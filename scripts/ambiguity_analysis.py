#!/usr/bin/env python3
"""Multi-gate ambiguity analysis (eval_specs.md Experiment C).

For every scored drift, record which of the five top-level gates fire INDEPENDENTLY --
i.e. before cascade precedence picks a single winner -- using the same predicate
functions `diagnose_decision_tree.classify_sig` calls internally:

  OCC        occ_shape(sig, floor)
  BRANCH     branch_tc(sig)
  SKIP       skip_sig(sig)
  REORDER    reorder_sig(sig)
  FREQUENCY  freq_sig(sig)

A drift is "ambiguous" when more than one gate fires (multi_gate). The cascade
(`classify_sig`, precedence order OCC > BRANCH > SKIP > REORDER > FREQUENCY) then
resolves it to one label; `resolved_correctly` checks that label against the benchmark's
LEAF_MEMBERS ground truth (same VALID_LABELS inverse as label_accuracy.py).

Outputs (eval_specs.md exact schemas):
  ambiguity_results.csv           per-drift gate firings + cascade outcome
  ambiguity_summary.csv           per-dataset ambiguity/resolution rates
  cross_dataset_consistency.csv   patterns present in both datasets: does the cascade's
                                   dominant label agree across Ostovar and Ceravolo?
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import diagnose_decision_tree as ddt

VALID_LABELS: dict[str, set[str]] = {}
for _leaf, _codes in ddt.LEAF_MEMBERS.items():
    for _c in _codes:
        VALID_LABELS.setdefault(_c, set()).add(_leaf)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", choices=["ostovar", "ceravolo", "both"], default="both")
    ap.add_argument("--logs-dir", type=Path, default=Path("cdrift-evaluation/EvaluationLogs/Ostovar"))
    ap.add_argument("--ceravolo-dir", type=Path, default=Path("cdrift-evaluation/EvaluationLogs/Ceravolo"))
    ap.add_argument("--results-csv", type=Path, default=Path("cdrift-evaluation/algorithm_results.csv"))
    ap.add_argument("--channels", default="support,confidence")
    ap.add_argument("--test", choices=["z", "chebyshev", "fisher"], default="z")
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--born-k", type=float, default=0.99)
    ap.add_argument("--tolerance", type=int, default=200)
    ap.add_argument("--tolerance-frac", type=float, default=0.15)
    ap.add_argument("--cer-tol", type=int, default=200)
    ap.add_argument("--sizes", default="1000", help="Ceravolo size filter (default: 1000, matching Experiment B)")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--span", type=int, default=0)
    ap.add_argument("--min-int-frac", type=float, default=0.0)


def gates_firing(sig, floor: float) -> list[str]:
    out = []
    if ddt.occ_shape(sig, floor):
        out.append("OCC")
    if ddt.branch_tc(sig):
        out.append("BRANCH")
    if ddt.skip_sig(sig):
        out.append("SKIP")
    if ddt.reorder_sig(sig):
        out.append("REORDER")
    if ddt.freq_sig(sig):
        out.append("FREQUENCY")
    return out


def analyze(ds: str, args, channels, floor: float) -> list[dict]:
    it = ddt.iter_ostovar(args, channels) if ds == "ostovar" else ddt.iter_ceravolo(args, channels)
    rows: list[dict] = []
    idx_counters: dict[str, int] = {}
    for code, sig in it:
        if sig is None:
            continue
        idx = idx_counters.get(code, 0)
        idx_counters[code] = idx + 1
        gates = gates_firing(sig, floor)
        cascade_label = ddt.classify_sig(sig, floor)
        valid = VALID_LABELS.get(code, set())
        primitive = "+".join(sorted(ddt.occ_shape(sig, floor))) if "OCC" in gates else ""
        rows.append({
            "dataset": ds, "drift_id": f"{ds}:{code}:{idx}", "pattern_code": code,
            "gates_firing": "|".join(gates), "gate_count": len(gates),
            "ambiguous": len(gates) > 1, "cascade_label": cascade_label,
            "cascade_correct": cascade_label in valid, "primitive_labels": primitive,
        })
    return rows


def write_summary(rows: list[dict], out_path: Path) -> list[dict]:
    summary_rows = []
    for ds in sorted({r["dataset"] for r in rows}) + ["ALL"]:
        rs = rows if ds == "ALL" else [r for r in rows if r["dataset"] == ds]
        if not rs:
            continue
        total = len(rs)
        single = sum(1 for r in rs if r["gate_count"] == 1)
        multi = sum(1 for r in rs if r["gate_count"] > 1)
        resolved_ok = sum(1 for r in rs if r["gate_count"] > 1 and r["cascade_correct"])
        summary_rows.append({
            "dataset": ds, "total_scored": total, "single_gate": single, "multi_gate": multi,
            "ambiguity_rate": round(multi / total, 4) if total else 0.0,
            "resolved_correctly": resolved_ok,
            "resolution_rate": round(resolved_ok / multi, 4) if multi else 0.0,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "total_scored", "single_gate", "multi_gate",
                                            "ambiguity_rate", "resolved_correctly", "resolution_rate"])
        w.writeheader()
        w.writerows(summary_rows)
    return summary_rows


def write_cross_dataset(rows: list[dict], out_path: Path) -> list[dict]:
    by_ds_code: dict[tuple[str, str], Counter] = {}
    for r in rows:
        by_ds_code.setdefault((r["dataset"], r["pattern_code"]), Counter())[r["cascade_label"]] += 1
    codes_by_ds = {ds: {c for (d, c) in by_ds_code if d == ds} for ds in ("ostovar", "ceravolo")}
    shared = sorted(codes_by_ds.get("ostovar", set()) & codes_by_ds.get("ceravolo", set()))

    out_rows = []
    for code in shared:
        ost_dist = by_ds_code.get(("ostovar", code), Counter())
        cer_dist = by_ds_code.get(("ceravolo", code), Counter())
        ost_dom = ost_dist.most_common(1)[0][0] if ost_dist else ""
        cer_dom = cer_dist.most_common(1)[0][0] if cer_dist else ""
        out_rows.append({
            "pattern_code": code,
            "ostovar_label_distribution": "|".join(f"{k}:{v}" for k, v in ost_dist.most_common()),
            "ceravolo_label_distribution": "|".join(f"{k}:{v}" for k, v in cer_dist.most_common()),
            "consistent": bool(ost_dom) and ost_dom == cer_dom,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pattern_code", "ostovar_label_distribution",
                                            "ceravolo_label_distribution", "consistent"])
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out-rows", type=Path, default=Path("results/ambiguity_results.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("results/ambiguity_summary.csv"))
    ap.add_argument("--out-cross", type=Path, default=Path("results/cross_dataset_consistency.csv"))
    args = ap.parse_args()

    if args.sizes:
        unknown = {s.strip() for s in args.sizes.split(",") if s.strip()} - set(ddt.SIZES)
        if unknown:
            ap.error(f"unknown --sizes {sorted(unknown)}; choose from {','.join(ddt.SIZES)}")
    channels = {c.strip().lower() for c in args.channels.split(",") if c.strip()}
    if not channels or not channels <= ddt.CHANNELS:
        ap.error(f"--channels must be a non-empty subset of {sorted(ddt.CHANNELS)}; got {sorted(channels)}")
    floor = 1.0 - args.born_k

    datasets = ["ostovar", "ceravolo"] if args.dataset == "both" else [args.dataset]
    rows: list[dict] = []
    for ds in datasets:
        rows += analyze(ds, args, channels, floor)

    if not rows:
        print("no scored drifts -- check --logs-dir/--ceravolo-dir/--sizes/--dataset", file=sys.stderr)
        return 2

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "drift_id", "pattern_code", "gates_firing",
                                            "gate_count", "ambiguous", "cascade_label", "cascade_correct",
                                            "primitive_labels"])
        w.writeheader()
        w.writerows(rows)

    summary_rows = write_summary(rows, args.out_summary)
    cross_rows = write_cross_dataset(rows, args.out_cross) if args.dataset == "both" else []

    print(f"Ambiguity analysis | dataset={args.dataset} sizes={args.sizes or 'all'} | K_born={args.born_k}\n")
    print(f"  {'dataset':10}{'scored':>8}{'single':>8}{'multi':>7}{'amb_rate':>10}{'resolved_ok':>13}{'resolution_rate':>17}")
    for s in summary_rows:
        print(f"  {s['dataset']:10}{s['total_scored']:>8}{s['single_gate']:>8}{s['multi_gate']:>7}"
              f"{s['ambiguity_rate']:>10.3f}{s['resolved_correctly']:>13}{s['resolution_rate']:>17.3f}")
    if cross_rows:
        print("\n  Cross-dataset consistency (patterns present in both):")
        print(f"    {'code':6}{'consistent':>11}   ostovar dist / ceravolo dist")
        for r in cross_rows:
            print(f"    {r['pattern_code']:6}{str(r['consistent']):>11}   {r['ostovar_label_distribution']} / {r['ceravolo_label_distribution']}")

    print(f"\nWrote {args.out_rows}, {args.out_summary}" + (f", {args.out_cross}" if cross_rows else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
