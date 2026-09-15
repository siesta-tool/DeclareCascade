#!/usr/bin/env python3
"""One-at-a-time detection-parameter sweeps (Experiment E, extended).

Sweeps four detection parameters, holding the other three at the anchor/default
configuration (xi=0.01, s=1, |B|=100, report=end, confidence=0.999), and writes one tidy
CSV of per-(sweep point, dataset) results that make_figures.py's make_fig5 consumes:

  xi      min_int_frac      relative-intensity confirmation threshold
  s       span              trailing-window width
  |B|     batch_size        batch size
  report  report            localisation policy (start/mid/end/adaptive/refine)

Reuses the production detection code path -- cdrift_approach.build_detector_series (per
batch size) and detect_drift_incremental.sequential_detect (per (batch size, span)) are
each computed ONCE and reused across every config that shares them, then
cdrift_approach.report_events (per config) applies the min_int_frac gate and report
policy. This is the same pipeline detect_changepoints() runs internally -- see
--verify-anchor, which cross-checks the shared-series path against detect_changepoints()
directly on every log.

Every swept config is FULLY specified: the |B| sweep pins span=1 explicitly rather than
letting the resolution rule re-derive it from the new batch count (see
detect_drift_incremental.resolve_config), so the |B| axis never silently moves s too.

Aggregation matches evaluate_cdrift.print_summary exactly: macro-mean of per-log F1 over
ALL logs (zero_division=0.0), and mean Avg-Lag over only the logs with >=1 TP -- computed
from raw floats, never from rounded CSV columns.

  python3 scripts/param_sweeps.py --limit 2                    # ~30s smoke test
  python3 scripts/param_sweeps.py --verify-anchor              # full run, ~12 min
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from evaluate_cdrift import F1_Score, get_avg_lag, getTP_FP, logpaths_with_changepoints

from cdrift_approach import (
    ORDER_TEMPLATES,
    build_detector_series,
    detect_changepoints,
    report_events,
)
from detect_drift_incremental import sequential_detect
from log_io import parse_traces

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = ("support", "confidence")
DATASETS_ORDER = ("Bose", "Ceravolo", "Ostovar")

ANCHOR = {"min_int_frac": 0.01, "span": 1, "batch_size": 100, "report": "end", "confidence": 0.999}

# axis label -> (config key it overrides, swept values)
AXES: dict[str, tuple[str, list]] = {
    "xi":     ("min_int_frac", [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]),
    "s":      ("span",         [1, 2, 3, 4, 5]),
    "B":      ("batch_size",   [25, 50, 100, 150, 200, 300]),
    "report": ("report",       ["start", "mid", "end", "adaptive", "refine"]),
}

FIELDS = ["sweep", "value", "dataset", "n_logs", "min_int_frac", "span", "batch_size", "report",
          "confidence", "is_anchor", "mean_f1", "mean_lag", "P", "R", "logs_with_tp",
          "mean_detections", "n_boundaries"]
ROW_FIELDS = ["sweep", "value", "cfg_key", "dataset", "log", "f1", "lag", "n_detections",
              "n_boundaries"]


def cfg_key(cfg: dict) -> str:
    return (f"B{cfg['batch_size']}_s{cfg['span']}_xi{cfg['min_int_frac']}"
            f"_{cfg['report']}_K{cfg['confidence']}")


def build_configs(axes: list[str]) -> tuple[list[tuple[str, object, dict, str]], dict[str, dict]]:
    """([(axis, value, cfg, key), ...] in figure order, {key: cfg} deduplicated for computation).

    Every cfg is a full copy of ANCHOR with exactly one field overridden -- in particular each
    |B| point pins span=1 explicitly (see module docstring). The anchor itself is shared by all
    four axes: it is computed once (one key) and appears once per axis in `points`, so all four
    curves cross at literally the same computed cell."""
    points: list[tuple[str, object, dict, str]] = []
    unique: dict[str, dict] = {}
    for axis in axes:
        key_name, values = AXES[axis]
        for v in values:
            cfg = dict(ANCHOR, **{key_name: v})
            k = cfg_key(cfg)
            unique.setdefault(k, cfg)
            points.append((axis, v, cfg, k))
    return points, unique


def plan_by_batch(unique: dict[str, dict]) -> dict[int, dict[int, list[str]]]:
    """{batch_size: {span: [cfg_key, ...]}} -- the loop skeleton that runs build_detector_series
    once per batch size and sequential_detect once per (batch size, span)."""
    plan: dict[int, dict[int, list[str]]] = {}
    for k, cfg in unique.items():
        plan.setdefault(cfg["batch_size"], {}).setdefault(cfg["span"], []).append(k)
    return plan


def limit_logs(logs: list[tuple[Path, list[int], str]], limit: int) -> list[tuple[Path, list[int], str]]:
    if not limit:
        return logs
    seen: dict[str, int] = defaultdict(int)
    out = []
    for path, known, source in logs:
        if seen[source] < limit:
            out.append((path, known, source))
            seen[source] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdrift-root", type=Path, default=ROOT / "cdrift-evaluation")
    ap.add_argument("--lag", type=int, default=200)
    ap.add_argument("--axes", default=",".join(AXES), help=f"comma-separated subset of {','.join(AXES)}")
    ap.add_argument("--datasets", default=",".join(DATASETS_ORDER))
    ap.add_argument("--limit", type=int, default=0, help="first N logs per dataset (smoke test)")
    ap.add_argument("--verify-anchor", action="store_true",
                    help="also run detect_changepoints() at the anchor on every log and assert "
                         "the shared-series sweep path agrees (adds ~1.5 min)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "param_sweeps.csv")
    ap.add_argument("--out-rows", type=Path, default=None,
                    help="optional per-(log, config) dump for debugging a specific cell")
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown_axes = set(axes) - set(AXES)
    if unknown_axes:
        ap.error(f"unknown --axes {sorted(unknown_axes)}; choose from {sorted(AXES)}")
    wanted_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    points, unique = build_configs(axes)
    plan = plan_by_batch(unique)
    anchor_key = cfg_key(ANCHOR)
    if anchor_key not in unique:
        # only happens if every requested axis omits the anchor value entirely
        unique[anchor_key] = dict(ANCHOR)

    logs = logpaths_with_changepoints(args.cdrift_root)
    logs = [row for row in logs if row[2] in wanted_datasets]
    logs = limit_logs(logs, args.limit)
    if not logs:
        print(f"no evaluation logs under {args.cdrift_root}/EvaluationLogs matching --datasets")
        return 2

    print(f"Parameter sweeps | axes={axes} | {len(unique)} unique configs over {len(points)} "
          f"sweep points | {len(logs)} logs | lag={args.lag}\n")

    # (cfg_key, dataset) -> one record per log: {f1, lag, n_det, testable}
    acc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    row_dump: list[dict] = []
    anchor_mismatches: list[str] = []

    for li, (path, known, source) in enumerate(logs, 1):
        traces = parse_traces(path, "concept:name")
        n = len(traces)
        anchor_detected_shared = None

        for B, by_span in plan.items():
            series = build_detector_series(traces, B) if n >= 4 else None
            nb = len(series.batches) if series is not None else 0
            for s, keys in by_span.items():
                cps = (sequential_detect(series, set(CHANNELS), "z", ANCHOR["confidence"], s,
                                         templates=ORDER_TEMPLATES)
                       if series is not None else [])
                testable = max(0, nb - 2 * s + 1) if series is not None else 0
                for k in keys:
                    cfg = unique[k]
                    detected = (report_events(traces, series, cps, cfg["min_int_frac"], cfg["report"])
                               if series is not None else [])
                    if k == anchor_key:
                        anchor_detected_shared = detected
                    f1 = float(F1_Score(detected, known, lag=args.lag, zero_division=0.0))
                    lag = float(get_avg_lag(detected, known, lag=args.lag))
                    tp, fp = getTP_FP(detected, known, lag=args.lag)
                    acc[(k, source)].append({"f1": f1, "lag": lag, "n_det": len(detected),
                                             "testable": testable, "tp": tp, "fp": fp,
                                             "n_known": len(known)})
                    if args.out_rows:
                        row_dump.append({"cfg_key": k, "dataset": source, "log": path.name,
                                         "f1": round(f1, 6),
                                         "lag": ("" if math.isnan(lag) else round(lag, 1)),
                                         "n_detections": len(detected), "n_boundaries": testable})
            series = None  # release before the next batch size

        if args.verify_anchor and anchor_detected_shared is not None:
            direct = detect_changepoints(traces, min_int_frac=ANCHOR["min_int_frac"],
                                         report=ANCHOR["report"], confidence=ANCHOR["confidence"],
                                         batch_size=ANCHOR["batch_size"], span=ANCHOR["span"])
            if direct != anchor_detected_shared:
                anchor_mismatches.append(f"{path.name}: shared={anchor_detected_shared} direct={direct}")

        if li % 25 == 0 or li == len(logs):
            print(f"  ...{li}/{len(logs)} logs")

    if anchor_mismatches:
        print(f"\n[FAIL] --verify-anchor: {len(anchor_mismatches)} log(s) disagree with detect_changepoints():")
        for m in anchor_mismatches[:10]:
            print(f"  {m}")
        return 1
    if args.verify_anchor:
        print(f"\n[ok] --verify-anchor: shared-series path agrees with detect_changepoints() on all {len(logs)} logs")

    rows = []
    for axis, value, cfg, k in points:
        for ds in DATASETS_ORDER:
            recs = acc.get((k, ds))
            if not recs:
                continue
            f1s = [r["f1"] for r in recs]                                    # ALL logs
            lags = [r["lag"] for r in recs if not math.isnan(r["lag"])]      # only logs with >=1 TP
            testable = [r["testable"] for r in recs]
            TP, FP, KNOWN = sum(r["tp"] for r in recs), sum(r["fp"] for r in recs), sum(r["n_known"] for r in recs)
            prec = TP / (TP + FP) if TP + FP else 0.0
            rec = TP / KNOWN if KNOWN else 0.0
            rows.append({
                "sweep": axis, "value": value, "dataset": ds, "n_logs": len(recs),
                "min_int_frac": cfg["min_int_frac"], "span": cfg["span"],
                "batch_size": cfg["batch_size"], "report": cfg["report"],
                "confidence": cfg["confidence"], "is_anchor": int(k == anchor_key),
                # 6 dp, not 4: the published numbers are quoted at 3 dp, and a mean of 0.9215253
                # stored as 0.9215 re-formats to 0.921 rather than 0.922 -- a silent double-round
                # that would make the paper's figure disagree with its own text.
                "mean_f1": round(sum(f1s) / len(f1s), 6),
                "mean_lag": ("" if not lags else round(sum(lags) / len(lags), 1)),
                "P": round(prec, 6), "R": round(rec, 6),
                "logs_with_tp": len(lags),
                "mean_detections": round(sum(r["n_det"] for r in recs) / len(recs), 2),
                "n_boundaries": int(round(statistics.median(testable))),
            })

    if not rows:
        print("no rows produced -- check --datasets/--axes")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")

    if args.out_rows and row_dump:
        args.out_rows.parent.mkdir(parents=True, exist_ok=True)
        with args.out_rows.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            w.writeheader()
            w.writerows(row_dump)
        print(f"Wrote {args.out_rows} ({len(row_dump)} rows)")

    print("\nAnchor rows (should read Ceravolo ~0.911/5.7 and Ostovar ~0.922/20.7 on the default grid):")
    for r in rows:
        if r["is_anchor"]:
            print(f"  sweep={r['sweep']:<7} {r['dataset']:10} F1={r['mean_f1']:.4f}  Avg-Lag={r['mean_lag']}")

    print(f"\n{'sweep':7}{'value':>8}{'dataset':>10}{'logs':>6}{'F1':>8}{'Avg-Lag':>9}{'TP-logs':>9}{'boundaries':>12}")
    for r in rows:
        print(f"{r['sweep']:7}{str(r['value']):>8}{r['dataset']:>10}{r['n_logs']:>6}"
              f"{r['mean_f1']:>8.3f}{str(r['mean_lag']):>9}{r['logs_with_tp']:>9}{r['n_boundaries']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
