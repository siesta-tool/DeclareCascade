#!/usr/bin/env python3
"""Evaluate our lean Declare-drift detector with the cdrift benchmark's own scoring.

F1-Score (TP via the LP assignment `assign_changepoints`) and Average Lag (mean |detected-actual|
over assigned pairs) from Adams et al., "An Experimental Evaluation of Process Concept Drift
Detection". The four scoring functions below are copied unchanged from
`cdrift-evaluation/cdrift/evaluation.py` (only isolated from that module's pm4py/matplotlib
imports). Ground-truth change points and the 1000-trace Ceravolo restriction replicate cdrift's
`get_logpaths_with_changepoints()`.

The functions are verbatim; the `zero_division=0.0` convention at the call site is ours. Upstream's
driver passes NaN and so drops logs with no true positive; we score those as F1 0.0. That is the
conservative direction (over the 151 committed rows 6 have TP=0: Ceravolo mean F1 is 0.911 here and
would be 0.976 if the undefined rows were dropped; Ostovar 0.870 vs 0.882), and it matches the
`.fillna(0.0)` `compare_to_cdrift.py` applies to the benchmark methods, so the head-to-head is
symmetric.

`--report` chooses where inside the firing batch the change point is placed: a batch edge
(start/mid/end), coarse-to-fine `refine`, or `adaptive` -- pick the edge by whichever neighbour the
firing batch resembles.

  python3 scripts/evaluate_cdrift.py            # all datasets, lag=200
  python3 scripts/evaluate_cdrift.py --min-int-frac 0   # recall-first (no confirmation)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from pulp import PULP_CBC_CMD, LpBinary, LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum

from cdrift_approach import detect_changepoints
from log_io import log_base_name, parse_traces

# Both default paths resolve against the repo root, so that running this from any cwd writes the
# same CSV that compare_to_cdrift.py (which does the same) reads back.
ROOT = Path(__file__).resolve().parent.parent

# cdrift scoring -- verbatim from cdrift/evaluation.py (Adams et al.)

def assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200):
    def buildProb_NoObjective(sense):
        prob = LpProblem("Changepoint_Assignment", sense)
        vars = LpVariable.dicts("x", (detected_changepoints, actual_changepoints), 0, 1, LpBinary)
        x = {(dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in actual_changepoints}
        for ap in actual_changepoints:
            prob += (lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                     f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}")
        for dp in detected_changepoints:
            prob += (lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                     f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}")
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                prob += (x[dp, ap] * abs(dp - ap) <= lag_window, f"Distance_Within_Lag_Window : {dp}_{ap}")
        return prob, x

    solver = PULP_CBC_CMD(msg=0)
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (lpSum(prob1_vars[dp, ap] for dp in detected_changepoints for ap in actual_changepoints),
              "Maximize number of assignments")
    prob1.solve(solver)
    num_tp = len([(dp, ap) for dp in detected_changepoints for ap in actual_changepoints
                  if prob1_vars[dp, ap].varValue == 1])
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (lpSum(prob2_vars[dp, ap] * pow(dp - ap, 2) for dp in detected_changepoints for ap in actual_changepoints),
              "Squared_Distances")
    prob2 += (lpSum(prob2_vars[dp, ap] for dp in detected_changepoints for ap in actual_changepoints) == num_tp,
              "Maximize Number of Assignments")
    prob2.solve(solver)
    return [(dp, ap) for dp in detected_changepoints for ap in actual_changepoints
            if prob2_vars[dp, ap].varValue == 1]


def getTP_FP(detected, known, lag, count_duplicate_detections=True):
    assignments = assign_changepoints(detected, known, lag_window=lag)
    TP = len(assignments)
    if count_duplicate_detections:
        FP = len(detected) - TP
    else:
        # kept for fidelity with upstream; this repo always uses the default (True)
        true_positive_candidates = [d for d in detected if any((k - lag <= d and d <= k + lag) for k in known)]
        FP = len(detected) - len(true_positive_candidates)
    return (TP, FP)


def F1_Score(detected, known, lag, zero_division=np.nan, count_duplicate_detections=True):
    TP, FP = getTP_FP(detected, known, lag, count_duplicate_detections)
    try:
        precision = TP / (TP + FP)
        recall = TP / len(known)
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return zero_division


def get_avg_lag(detected_changepoints, actual_changepoints, lag=200):
    assignments = assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag)
    avg_lag = 0
    for (dc, ap) in assignments:
        avg_lag += abs(dc - ap)
    try:
        return avg_lag / len(assignments)
    except ZeroDivisionError:
        return np.nan

# datasets -- replicate cdrift's get_logpaths_with_changepoints()

def logpaths_with_changepoints(root: Path):
    out = []
    bose = root / "EvaluationLogs" / "Bose" / "bose_log.xes.gz"
    if bose.exists():
        out.append((bose, [1199, 2399, 3599, 4799], "Bose"))
    cer = root / "EvaluationLogs" / "Ceravolo"
    if cer.exists():
        for item in sorted(cer.iterdir()):
            parts = item.stem.split("_")
            # cdrift evaluates only the 1000-trace Ceravolo logs; each is baseline for its first
            # half and drifted for its second, so the change point is n//2 - 1
            if len(parts) >= 4 and parts[3] == "1000":
                out.append((item, [1000 // 2 - 1], "Ceravolo"))
    ost = root / "EvaluationLogs" / "Ostovar"
    if ost.exists():
        for item in sorted(ost.iterdir()):
            out.append((item, [999, 1999], "Ostovar"))
    return out


def print_summary(per_source: dict[str, list]) -> None:
    print(f"{'dataset':10}{'logs':>6}{'mean F1':>9}{'mean Avg-Lag':>14}{'logs w/ TP':>12}")
    for source in ("Bose", "Ceravolo", "Ostovar"):
        rs = per_source.get(source)
        if not rs:
            continue
        f1s = [r[0] for r in rs]
        lags = [r[1] for r in rs if not np.isnan(r[1])]          # Avg-Lag is defined only when >=1 TP
        mean_lag = f"{sum(lags) / len(lags):.1f}" if lags else "-"
        print(f"{source:10}{len(rs):>6}{sum(f1s) / len(f1s):>9.3f}{mean_lag:>14}{f'{len(lags)}/{len(rs)}':>12}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdrift-root", type=Path, default=ROOT / "cdrift-evaluation")
    ap.add_argument("--lag", type=int, default=200)
    ap.add_argument("--min-int-frac", type=float, default=0.01, help="relative intensity confirmation (0 = recall-first)")
    ap.add_argument("--report", choices=["start", "mid", "end", "refine", "adaptive"], default="end", help="change-point location within the firing batch")
    ap.add_argument("--confidence", type=float, default=0.999, help="statistical confidence K for the sequential detector")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="override the auto batch size |B| (0 = auto, ~100 cases/batch)")
    ap.add_argument("--span", type=int, default=0,
                    help="override the trailing-window width s (0 = auto; note that overriding "
                         "--batch-size alone re-derives s from the new batch count)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "cdrift-eval.csv")
    args = ap.parse_args()

    default_out = ROOT / "results" / "cdrift-eval.csv"
    if (args.batch_size or args.span) and args.out == default_out:
        ap.error("--batch-size/--span change the configuration; pass an explicit --out so the "
                 "headline results/cdrift-eval.csv (read back by compare_to_cdrift.py and "
                 "make_figures.py) is not silently overwritten with a non-default run")

    rows = []
    per_source: dict[str, list] = {}
    logs = logpaths_with_changepoints(args.cdrift_root)
    if not logs:
        print(f"no evaluation logs under {args.cdrift_root}/EvaluationLogs "
              "-- see README for obtaining the cdrift benchmark", file=sys.stderr)
        return 2
    print(f"Our lean Declare-drift detector | cdrift scoring (F1 + Average Lag) | lag={args.lag} | "
          f"min_int_frac={args.min_int_frac} | confidence={args.confidence} | "
          f"B={args.batch_size or 'auto'} s={args.span or 'auto'} | {len(logs)} logs\n")
    for path, known, source in logs:
        traces = parse_traces(path, "concept:name")
        detected = detect_changepoints(traces, min_int_frac=args.min_int_frac, report=args.report,
                                        confidence=args.confidence, batch_size=args.batch_size,
                                        span=args.span)
        f1 = F1_Score(detected, known, lag=args.lag, zero_division=0.0)
        lag = get_avg_lag(detected, known, lag=args.lag)
        rows.append({"Algorithm": "DeclareTree", "Log Source": source, "Log": log_base_name(path),
                     "Detected Changepoints": detected, "Actual Changepoints for Log": known,
                     "F1-Score": round(float(f1), 4), "Average Lag": ("" if np.isnan(lag) else round(float(lag), 1))})
        per_source.setdefault(source, []).append((f1, lag, len(detected), len(known)))

    print_summary(per_source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # report the path repo-relative when it is inside the repo, so stdout does not depend on cwd
    shown = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out
    print(f"\nWrote per-log results -> {shown}")
    print("Columns match cdrift algorithm_results.csv (Algorithm, Log Source, Log, Detected/Actual Changepoints, F1-Score, Average Lag).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
