#!/usr/bin/env python3
"""Detection channel ablation (eval_specs.md Experiment E.2.3).

Runs `cdrift_approach.detect_changepoints` on every benchmark log for three channel
configurations -- support only, confidence only, and both (the default) -- and scores
each with the cdrift benchmark's own F1-Score / Average Lag (evaluate_cdrift.py,
verbatim from Adams et al.).

  python3 scripts/channel_ablation.py --out results/channel_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cdrift_approach import detect_changepoints
from evaluate_cdrift import F1_Score, get_avg_lag, getTP_FP, logpaths_with_changepoints
from log_io import parse_traces

ROOT = Path(__file__).resolve().parent.parent

CHANNEL_CONFIGS = {
    "support": ("support",),
    "confidence": ("confidence",),
    "support+confidence": ("support", "confidence"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdrift-root", type=Path, default=ROOT / "cdrift-evaluation")
    ap.add_argument("--lag", type=int, default=200)
    ap.add_argument("--min-int-frac", type=float, default=0.01)
    ap.add_argument("--report", choices=["start", "mid", "end", "refine", "adaptive"], default="end")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "channel_ablation.csv")
    args = ap.parse_args()

    logs = logpaths_with_changepoints(args.cdrift_root)
    if not logs:
        print(f"no evaluation logs under {args.cdrift_root}/EvaluationLogs")
        return 2

    # parse traces once per log, reuse across the three channel configs
    parsed = [(path, known, source, parse_traces(path, "concept:name")) for path, known, source in logs]

    rows = []
    print(f"Channel ablation | {len(logs)} logs | lag={args.lag} min_int_frac={args.min_int_frac} report={args.report}\n")
    for label, channels in CHANNEL_CONFIGS.items():
        per_source: dict[str, list] = {}
        for path, known, source, traces in parsed:
            detected = detect_changepoints(traces, min_int_frac=args.min_int_frac, channels=channels, report=args.report)
            f1 = F1_Score(detected, known, lag=args.lag, zero_division=0.0)
            lag = get_avg_lag(detected, known, lag=args.lag)
            tp, fp = getTP_FP(detected, known, lag=args.lag)
            per_source.setdefault(source, []).append((f1, lag, tp, fp, len(known)))
        for source, rs in per_source.items():
            f1s = [r[0] for r in rs]
            lags = [r[1] for r in rs if not np.isnan(r[1])]
            mean_lag = sum(lags) / len(lags) if lags else float("nan")
            TP, FP, KNOWN = sum(r[2] for r in rs), sum(r[3] for r in rs), sum(r[4] for r in rs)
            prec = TP / (TP + FP) if TP + FP else 0.0
            rec = TP / KNOWN if KNOWN else 0.0
            rows.append({"dataset": source, "channels": label, "logs": len(rs),
                         "P": round(prec, 4), "R": round(rec, 4), "F1": round(sum(f1s) / len(f1s), 4),
                         "Avg-Lag": ("" if not lags else round(mean_lag, 1))})
        print(f"  {label}:")
        for source in ("Bose", "Ceravolo", "Ostovar"):
            r = next((r for r in rows if r["dataset"] == source and r["channels"] == label), None)
            if r:
                print(f"    {source:10} logs={r['logs']:<4} F1={r['F1']:.3f}  Avg-Lag={r['Avg-Lag']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "channels", "logs", "P", "R", "F1", "Avg-Lag"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
