"""Lean Declare-drift detector in the cdrift approach interface.

Threshold-free: the only two free parameters are the confidence K and the relative-intensity
fraction `min_int_frac`. No magnitude thresholds anywhere.

Input : an ordered list of traces (each a tuple/list of activity names), or a PM4Py EventLog
        via `detect(log)`.
Output: a list of trace-index change points -- exactly what cdrift's
        `evaluation.F1_Score(detected, known, lag)` / `get_avg_lag(...)` expect.

Pipeline: per-batch ChainResponse/Response support+confidence series -> sequential
two-proportion z-test with forgetting (recall-first) -> cluster boundaries into events ->
relative intensity confirmation (keep events shifting >= `min_int_frac` of the tracked
order-relations, a fraction that transfers across process densities).
"""

from __future__ import annotations

from batch_support_series import build_series
from detect_drift_incremental import Batch, Series, resolve_config, sequential_detect

# The two relation families the detector measures (batch_support_series naming) and the templates
# sequential_detect is restricted to firing on (Series naming). Named once so a caller building
# its own Series (e.g. a parameter sweep reusing build_detector_series across configs) cannot
# silently drift from what detect_changepoints itself measures.
SERIES_FAMILIES = {"chainresponse", "response"}
ORDER_TEMPLATES = {"ChainResponse", "Response"}


def _refine_location(traces, lo_case: int, hi_case: int, fine: int = 20) -> int:
    """Coarse-to-fine: within a local case window, find the sub-batch boundary where the drift's
    relations jump. Each relation is weighted by how much it flips across the whole window
    (w_r = |rate(left half) - rate(right half)|), so stable and noisy relations contribute ~0.
    Returns the absolute case index."""
    sub = traces[lo_case:hi_case]
    if len(sub) < 3 * fine:
        return lo_case + len(sub) // 2
    _, manifest, support, _, _ = build_series(sub, fine, SERIES_FAMILIES)
    ncases = [int(x["n_cases"]) for x in manifest]
    nfb = len(manifest)
    order = [r for r in support if r[0] in ORDER_TEMPLATES]
    total = sum(ncases)
    half = nfb // 2
    ln_h, rn_h = sum(ncases[:half]) or 1, sum(ncases[half:]) or 1
    # per-relation flip magnitude across the window -> weight
    w = {r: abs(sum(support[r][:half]) / ln_h - sum(support[r][half:]) / rn_h) for r in order}
    changed = [r for r in order if w[r] > 0]
    if not changed:
        return lo_case + total // 2

    def score_at(k: int) -> float:
        ln, rn = sum(ncases[:k]), sum(ncases[k:])
        if ln == 0 or rn == 0:
            return -1.0                                # degenerate split: never wins
        return sum(w[r] * abs(sum(support[r][:k]) / ln - sum(support[r][k:]) / rn) for r in changed)

    # seed at the centre so equal-scoring splits break toward it rather than toward k=1
    best_k, best = half, score_at(half)
    for k in range(1, nfb):
        score = score_at(k)
        if score > best:
            best, best_k = score, k
    return lo_case + sum(ncases[:best_k])


def build_detector_series(traces, batch_size: int) -> Series | None:
    """The detector's per-batch two-channel (support+confidence) series at one batch size, or
    None when the log yields no batches (e.g. too few traces for `batch_size`).

    Split out of detect_changepoints so a parameter sweep can build this ONCE per batch size and
    reuse it across every (span, min_int_frac, report) config that shares it -- build_series is
    the dominant cost of a detection pass (>90% of per-log runtime; see runtime_benchmark.py)."""
    _, manifest, support, fulf, act = build_series(traces, batch_size, SERIES_FAMILIES)
    if not manifest:
        return None
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"]))
               for x in manifest]
    return Series(batches, support, fulf, act)


def report_events(traces, series: Series, cps, min_int_frac: float = 0.01,
                  report: str = "end") -> list[int]:
    """Cluster raw change-point firings into events, apply the relative-intensity confirmation
    gate, and place each surviving event at a case index per `report`. Returns sorted, deduplicated
    case indices -- exactly detect_changepoints' return value.

    This is pure post-processing over an already-completed detection pass (`series`, `cps`): it
    never changes WHICH batches fired, only which of them survive confirmation and where they are
    reported. A parameter sweep over min_int_frac or report can therefore call this repeatedly
    against one shared (series, cps) pair. It must stay side-effect-free -- it reads `series` and
    `cps` and mutates neither.

    `report` chooses where in the firing batch to place the change point. The detector locates
    the *batch* that straddles the drift; a fixed edge can't be optimal for every dataset (the
    drift may fall in the firing batch or the one before). Options:
      start / mid / end -- batch edges; free and dataset-dependent.
      refine            -- coarse-to-fine: local scan of the flagged neighbourhood for the
                           actual support jump, giving sub-batch lag on all datasets.
      adaptive          -- does the firing batch resemble its PRE neighbour (drift at its end)
                           or its POST neighbour (drift already happened, so at its start)?
                           Report the matching edge."""
    batches, support = series.batches, series.support
    nb = len(batches)
    order_rels = [r for r in (set(series.support) | set(series.fulfillments)) if r[0] in ORDER_TEMPLATES]
    thr = min_int_frac * (len(order_rels) or 1)

    def _rate(r, i):                                  # r is always in order_rels, hence in support
        n = batches[i].n_cases
        return (support[r][i] / n) if n else 0.0

    def _dist(i, j):                                  # order-profile distance between two batches
        return sum(abs(_rate(r, i) - _rate(r, j)) for r in order_rels)

    def case_of(b: int) -> int:
        s, e = batches[b].start_case, batches[b].end_case
        if report == "refine":
            lo = batches[max(0, b - 1)].start_case
            hi = batches[min(nb - 1, b + 1)].end_case + 1
            return _refine_location(traces, lo, hi)
        if report == "adaptive":
            has_prev, has_next = b - 1 >= 0, b + 1 < nb
            if has_prev and has_next:
                return s if _dist(b, b + 1) < _dist(b, b - 1) else e
            return s if (has_next and not has_prev) else e
        if report == "start":
            return s
        if report == "mid":
            return (s + e) // 2
        if report == "end":
            return e
        raise ValueError(f"unknown report mode: {report!r}")

    # Cluster adjacent boundaries into events. The reported batch is the cluster's PEAK-intensity
    # member, not its first: the gate below tests the peak, so the location has to be the peak too.
    # Anchoring on the first member let a weak adjacent firing drag the change point a batch early.
    events: list[list[int]] = []   # [peak_boundary_batch, peak_intensity]
    prev_b = None
    for c in sorted(cps, key=lambda c: c.boundary):
        if prev_b is not None and c.boundary - prev_b <= 1:
            if c.intensity > events[-1][1]:       # strict >: ties keep the earlier batch
                events[-1] = [c.boundary, c.intensity]
        else:
            events.append([c.boundary, c.intensity])
        prev_b = c.boundary
    return sorted({case_of(b) for b, inten in events if inten >= thr})


def detect_changepoints(traces, min_int_frac: float = 0.01, test: str = "z",
                        confidence: float = 0.999, channels=("support", "confidence"),
                        report: str = "end", batch_size: int | None = None,
                        span: int | None = None) -> list[int]:
    """Return trace indices (case positions) of confirmed change points.

    `batch_size` / `span` override the auto_config heuristic (None or 0 = auto). Overriding
    `batch_size` ALONE re-derives `span` from the new batch count -- see
    detect_drift_incremental.resolve_config. A sweep over one of them must therefore pin BOTH
    explicitly, or it moves two parameters at once.

    See `report_events` for the `report` (localisation) options.
    """
    n = len(traces)
    if n < 4:
        return []
    batch_size, span = resolve_config(n, batch_size, span)
    series = build_detector_series(traces, batch_size)
    if series is None:
        return []
    cps = sequential_detect(series, set(channels), test, confidence, span, templates=ORDER_TEMPLATES)
    return report_events(traces, series, cps, min_int_frac, report)


def detect(log, activity_key: str = "concept:name", **kwargs) -> list[int]:
    """cdrift-style entry point: takes a PM4Py EventLog, returns trace-index change points."""
    traces = [tuple(ev.get(activity_key, "<missing>") for ev in trace) for trace in log]
    return detect_changepoints(traces, **kwargs)
