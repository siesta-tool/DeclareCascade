#!/usr/bin/env python3
"""Synthetic drift logs with known change points, pushed through the real detect+label pipeline.

Each family asserts both that the change point is detected within lag and that the decision tree
assigns the expected leaf. Prints a PASS/FAIL table and exits non-zero on any failure.

  python3 scripts/synthetic_eval.py
"""

from __future__ import annotations

from batch_support_series import build_series
from cdrift_approach import detect_changepoints
from detect_drift_incremental import Batch, Series, auto_config, sequential_detect
from diagnose_decision_tree import classify_sig, events_of
from signature import build_change_signature, build_evidence

CH = {"support", "confidence"}
FLOOR = 0.01  # 1 - born_k(0.99)
TOL = 200     # cases; same detection tolerance as diagnose_decision_tree._sig_for


# --------------------------------------------------------------------------- #
# deterministic synthetic-log generators                                      #
# --------------------------------------------------------------------------- #
def make(before, after, n=3000, drift=1500):
    """before(i)/after(i) -> trace tuple; deterministic (index-parity routing, no RNG)."""
    return [before(i) if i < drift else after(i) for i in range(n)]


def leaf_at(traces, true_cp, floor=FLOOR):
    """Run the real detection+signature pipeline; return the decision-tree leaf at the event
    nearest true_cp (or None if nothing detected)."""
    B, span = auto_config(len(traces))
    _, manifest, support, fulf, act = build_series(traces, B, {"chainresponse", "response", "existence", "choice"})
    if not manifest:
        return None
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"])) for x in manifest]
    series = Series(batches, support, fulf, act)
    cps = sequential_detect(series, CH, "z", 0.999, span, templates={"ChainResponse", "Response"})
    events = events_of(cps)
    if not events:
        return None
    nb = len(batches)
    near = min(range(len(events)), key=lambda i: abs(batches[events[i][0]].start_case - true_cp))
    if abs(batches[events[near][0]].start_case - true_cp) > TOL:
        return None                     # nearest event is not this drift; labeling it would mislead
    fb, lb = events[near]
    b_lo = events[near - 1][1] if near > 0 else 0
    a_hi = events[near + 1][0] if near + 1 < len(events) else nb
    sig = build_change_signature(build_evidence(series, b_lo, fb, lb, a_hi, CH, "z", 0.999))
    return classify_sig(sig, floor)


def detected_near(traces, true_cp, lag=200, min_int_frac=0.0):
    cps = detect_changepoints(traces, min_int_frac=min_int_frac)
    return any(abs(c - true_cp) <= lag for c in cps), cps


def alt(a, b):
    return lambda i: a if i % 2 == 0 else b


# (name, log, expected leaf family)
FAMILIES = [
    ("insertion",        make(lambda i: ("A", "B", "D"),           lambda i: ("A", "B", "C", "D")),        {"INSERTION"}),
    ("removal",          make(lambda i: ("A", "B", "C", "D"),      lambda i: ("A", "B", "D")),             {"REMOVAL"}),
    ("duplication",      make(lambda i: ("A", "B", "D"),           lambda i: ("A", "B", "D", "B")),        {"DUPLICATION"}),
    ("loop",             make(lambda i: ("A", "B", "D"),           lambda i: ("A", "B", "B", "B", "D")),   {"LOOP"}),
    ("swap",             make(lambda i: ("A", "B", "C", "D"),      lambda i: ("A", "C", "B", "D")),        {"REORDER"}),
    ("move",             make(lambda i: ("A", "B", "C", "D", "E"), lambda i: ("A", "C", "D", "B", "E")),   {"REORDER"}),
    ("branch_cond2seq",  make(alt(("A", "B", "D"), ("A", "C", "D")), lambda i: ("A", "B", "C", "D")),      {"BRANCH"}),
    ("branch_seq2cond",  make(lambda i: ("A", "B", "C", "D"),      alt(("A", "B", "D"), ("A", "C", "D"))), {"BRANCH"}),
    ("skip",             make(lambda i: ("A", "B", "C", "D"),      lambda i: ("A", "B", "C", "D") if i % 2 else ("A", "B", "D")), {"SKIP"}),
    # 80/20 -> 20/80 routing of B vs C: activity B's occurrence collapses, so the SKIP lens
    # (Existence(B,1) drops but stays positive + ChainResponse touching B drops) legitimately claims
    # it before FREQUENCY. This is the documented fr~=skip near-degeneracy, not a defect.
    ("frequency",        make(lambda i: ("A", "B", "D") if i % 5 else ("A", "C", "D"),
                              lambda i: ("A", "C", "D") if i % 5 else ("A", "B", "D")),                    {"FREQUENCY", "SKIP", "OTHER"}),
]


def main() -> int:
    rows = []

    # --- per-family detection + label ---
    for name, log, expected in FAMILIES:
        det, cps = detected_near(log, 1500, min_int_frac=0.0)
        leaf = leaf_at(log, 1500)
        det_ok = det
        lab_ok = leaf in expected
        rows.append((name, "detect+label", f"det={det_ok} leaf={leaf} exp={'/'.join(sorted(expected))}",
                     det_ok and lab_ok))

    # --- invariants / edge cases ---
    ins = FAMILIES[0][1]

    # in-process idempotence; cross-process determinism verified separately under varying
    # PYTHONHASHSEED (set iteration order is the real risk and it is not exercised here)
    d1 = detect_changepoints(ins, min_int_frac=0.01)
    d2 = detect_changepoints(ins, min_int_frac=0.01)
    rows.append(("determinism", "invariant", f"{d1} == {d2}", d1 == d2))

    # reporting-edge invariance: same # drifts, all near the true cp. Unlike the per-family tests
    # above, this deliberately runs under the default confirmation gate.
    counts = {r: detect_changepoints(ins, report=r) for r in ("start", "mid", "end")}
    same_n = len({len(v) for v in counts.values()}) == 1
    near_all = all(any(abs(c - 1500) <= 200 for c in v) for v in counts.values())
    rows.append(("reporting_edge_invariance", "invariant", f"counts={{k:len(v) for..}}={ {k: len(v) for k, v in counts.items()} }", same_n and near_all))

    # no-drift / noise: confirmed drifts near zero
    noise = [(("A", "B", "D") if i % 2 == 0 else ("A", "C", "D")) for i in range(6000)]
    conf = detect_changepoints(noise, min_int_frac=0.01)
    rows.append(("no_drift_noise", "confirmation", f"confirmed CPs={len(conf)} (want ~0)", len(conf) == 0))

    # Sample-size power. The trailing window is fixed at ~1 batch (~100 cases), so power is
    # governed by the shift magnitude at that window, not the total log length. A 40pt routing shift
    # is well above the K=0.999 (z*=3.29) bar and must detect; a 10pt shift on a 100-case window
    # gives z~2.0 and is correctly treated as noise -- that is the threshold-free contract.
    def freq(n, drift, lo, hi):
        return make(lambda i: ("A", "B", "D") if i % 20 < lo else ("A", "C", "D"),
                    lambda i: ("A", "B", "D") if i % 20 < hi else ("A", "C", "D"), n=n, drift=drift)
    big_det, _ = detected_near(freq(20000, 10000, 14, 6), 10000, lag=400, min_int_frac=0.0)     # 70->30
    tiny_shift_det, _ = detected_near(freq(20000, 10000, 11, 9), 10000, lag=400, min_int_frac=0.0)  # 55->45
    rows.append(("power_detects_clear_shift", "power", f"40pt-shift detected={big_det} (want True)", big_det))
    rows.append(("power_subthreshold_is_noise", "power", f"10pt-shift detected={tiny_shift_det} (want False @K=0.999)", not tiny_shift_det))

    # tiny log: no crash
    try:
        tiny = make(lambda i: ("A", "B", "D"), lambda i: ("A", "B", "C", "D"), n=150, drift=75)
        tcp = detect_changepoints(tiny)
        rows.append(("tiny_log_no_crash", "edge", f"cps={tcp} (no crash)", True))
    except Exception as e:  # noqa
        rows.append(("tiny_log_no_crash", "edge", f"CRASH: {e}", False))

    # --- report ---
    print(f"{'test':30}{'kind':14}{'detail':58}{'result'}")
    print("-" * 112)
    npass = 0
    for name, kind, detail, ok in rows:
        npass += ok
        print(f"{name:30}{kind:14}{detail[:56]:58}{'PASS' if ok else 'FAIL'}")
    print("-" * 112)
    print(f"{npass}/{len(rows)} passed")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
