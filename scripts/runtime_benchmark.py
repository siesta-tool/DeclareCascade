#!/usr/bin/env python3
"""Runtime and retrospective-behaviour benchmark (eval_specs.md Experiment F).

Instruments the pipeline with `time.perf_counter` at five breakpoints, per benchmark log:

  1. batch      -- build_series(): batch construction + Declare (support/fulfillment) measurement
  2. detect     -- sequential_detect(): statistical testing + change-point detection
  3. cluster    -- event clustering + the dual confirmation gate (intensity >= xi OR structural label)
  4. signature  -- build_evidence() + build_change_signature() for each confirmed event
  5. label      -- classify_sig() cascade labelling for each confirmed event

Writes runtime_results.csv (per log) and runtime_summary.csv (median/p95 per dataset,
plus throughput). Retrospective delay (delay_batches = trailing window width s) is a
structural property, not a measured one -- printed once, not benchmarked per drift.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from batch_support_series import build_series
from detect_drift_incremental import Batch, Series, auto_config, sequential_detect
from diagnose_decision_tree import classify_sig
from evaluate_cdrift import logpaths_with_changepoints
from log_io import log_base_name, parse_traces
from signature import build_change_signature, build_evidence

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = {"support", "confidence"}


def _n_order(support, fulf) -> int:
    return sum(1 for r in (set(support) | set(fulf)) if r[0] in ("ChainResponse", "Response"))


def benchmark_log(path: Path, source: str, confidence: float, min_int_frac: float, born_k: float) -> dict:
    traces = parse_traces(path, "concept:name")
    n_cases = len(traces)
    n_activities = len({a for tr in traces for a in tr})

    t0 = time.perf_counter()
    B, span = auto_config(n_cases)
    _, manifest, support, fulf, act = build_series(traces, B, {"chainresponse", "response", "existence", "choice"})
    t_batch = time.perf_counter() - t0

    n_relations = len(set(support) | set(fulf))
    n_batches = len(manifest)
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"])) for x in manifest]
    series = Series(batches, support, fulf, act)

    t0 = time.perf_counter()
    cps = sequential_detect(series, CHANNELS, "z", confidence, span, templates={"ChainResponse", "Response"})
    t_detect = time.perf_counter() - t0

    t0 = time.perf_counter()
    events: list[list[int]] = []
    for c in sorted(cps, key=lambda c: c.boundary):
        if events and c.boundary - events[-1][1] <= 1:
            events[-1][1] = c.boundary
            events[-1][2] = max(events[-1][2], c.intensity)
        else:
            events.append([c.boundary, c.boundary, c.intensity])
    N = _n_order(support, fulf)
    thr = min_int_frac * max(1, N)
    confirmed_idx = [i for i, (fb, lb, inten) in enumerate(events) if inten >= thr]
    t_cluster = time.perf_counter() - t0

    floor = 1.0 - born_k
    t_sig_total = 0.0
    t_lab_total = 0.0
    for i in confirmed_idx:
        fb, lb, _inten = events[i]
        b_lo = events[i - 1][1] if i > 0 else 0
        a_hi = events[i + 1][0] if i + 1 < len(events) else len(batches)
        t0 = time.perf_counter()
        sig = build_change_signature(build_evidence(series, b_lo, fb, lb, a_hi, CHANNELS, "z", confidence))
        t_sig_total += time.perf_counter() - t0
        t0 = time.perf_counter()
        classify_sig(sig, floor)
        t_lab_total += time.perf_counter() - t0

    t_total = t_batch + t_detect + t_cluster + t_sig_total + t_lab_total
    return {
        "dataset": source, "log_name": log_base_name(path), "n_cases": n_cases,
        "n_activities": n_activities, "n_relations": n_relations, "n_batches": n_batches,
        "t_batch_ms": round(t_batch * 1000, 3), "t_detect_ms": round(t_detect * 1000, 3),
        "t_cluster_ms": round(t_cluster * 1000, 3), "t_confirm_ms": round(t_cluster * 1000, 3),
        "t_signature_ms": round(t_sig_total * 1000, 3), "t_label_ms": round(t_lab_total * 1000, 3),
        "t_total_ms": round(t_total * 1000, 3),
    }


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdrift-root", type=Path, default=ROOT / "cdrift-evaluation")
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--min-int-frac", type=float, default=0.01)
    ap.add_argument("--born-k", type=float, default=0.99)
    ap.add_argument("--out-rows", type=Path, default=ROOT / "results" / "runtime_results.csv")
    ap.add_argument("--out-summary", type=Path, default=ROOT / "results" / "runtime_summary.csv")
    args = ap.parse_args()

    logs = logpaths_with_changepoints(args.cdrift_root)
    if not logs:
        print(f"no evaluation logs under {args.cdrift_root}/EvaluationLogs")
        return 2

    print(f"Runtime benchmark | {len(logs)} logs | K={args.confidence} xi={args.min_int_frac} K_born={args.born_k}\n")
    rows = []
    for path, known, source in logs:
        rows.append(benchmark_log(path, source, args.confidence, args.min_int_frac, args.born_k))

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "log_name", "n_cases", "n_activities", "n_relations", "n_batches",
              "t_batch_ms", "t_detect_ms", "t_cluster_ms", "t_confirm_ms", "t_signature_ms",
              "t_label_ms", "t_total_ms"]
    with args.out_rows.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        total_cases = sum(r["n_cases"] for r in rs)
        total_s = sum(r["t_total_ms"] for r in rs) / 1000.0
        summary_rows.append({
            "dataset": ds, "n_logs": len(rs),
            "median_batch_ms": round(statistics.median(r["t_batch_ms"] for r in rs), 3),
            "p95_batch_ms": round(_pctl([r["t_batch_ms"] for r in rs], 0.95), 3),
            "median_detect_ms": round(statistics.median(r["t_detect_ms"] for r in rs), 3),
            "p95_detect_ms": round(_pctl([r["t_detect_ms"] for r in rs], 0.95), 3),
            "median_label_ms": round(statistics.median(r["t_label_ms"] for r in rs), 3),
            "p95_label_ms": round(_pctl([r["t_label_ms"] for r in rs], 0.95), 3),
            "median_total_ms": round(statistics.median(r["t_total_ms"] for r in rs), 3),
            "p95_total_ms": round(_pctl([r["t_total_ms"] for r in rs], 0.95), 3),
            "throughput_cases_per_sec": round(total_cases / total_s, 1) if total_s else 0.0,
        })

    with args.out_summary.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "n_logs", "median_batch_ms", "p95_batch_ms",
                                            "median_detect_ms", "p95_detect_ms", "median_label_ms",
                                            "p95_label_ms", "median_total_ms", "p95_total_ms",
                                            "throughput_cases_per_sec"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"  {'dataset':10}{'logs':>6}{'med batch':>11}{'med detect':>12}{'med label':>11}{'med total':>11}{'throughput/s':>14}")
    for s in summary_rows:
        print(f"  {s['dataset']:10}{s['n_logs']:>6}{s['median_batch_ms']:>11.2f}{s['median_detect_ms']:>12.2f}"
              f"{s['median_label_ms']:>11.3f}{s['median_total_ms']:>11.2f}{s['throughput_cases_per_sec']:>14.1f}")

    print("\nRetrospective delay (structural, not measured): delay_batches = trailing window s "
          "(auto_config: s=2 if #batches<=6 else 1); delay_cases = s * batch_size.")
    print(f"\nWrote {args.out_rows} and {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
