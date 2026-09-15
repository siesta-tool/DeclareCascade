#!/usr/bin/env python3
"""Ambiguity-aware label scorer (eval_specs.md Experiment B.4).

For every scored drift (detected by the pipeline and matched to a ground-truth change
point within tolerance -- exactly what `diagnose_decision_tree.iter_ostovar` /
`iter_ceravolo` already yield), classify DeclareCascade's structural label into one of
four categories:

  correct_unambiguous  -- the benchmark pattern has exactly one valid DeclareCascade
                           label (LEAF_MEMBERS), and the cascade returns it.
  correct_resolved     -- the pattern maps to multiple valid labels (e.g. `cm` -> {BRANCH,
                           REORDER}), and the cascade returns one of them.
  incorrect            -- the cascade returns a label outside the valid set.
  unsupported          -- the cascade returns OTHER for a pattern with no valid label at
                           all (`pl`, the known limitation, or an unmapped code like `sre`).

This script only adds the ambiguity bookkeeping and CSV outputs; the detection, signature
and classification logic are reused unchanged from diagnose_decision_tree.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import diagnose_decision_tree as ddt

# Inverse of LEAF_MEMBERS: benchmark pattern code -> set of DeclareCascade labels that
# count as correct for it. A code absent here (e.g. `pl`) has no valid label -> only
# OTHER is "expected".
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
    ap.add_argument("--min-int-frac", type=float, default=0.0,
                     help="labelling uses full-regime windows over ALL detected events, so this stays 0 by default")


def iter_drifts(args, channels):
    """Yield (dataset, pattern_code, signature-or-None) across the requested dataset(s)."""
    if args.dataset in ("ostovar", "both"):
        for code, sig in ddt.iter_ostovar(args, channels):
            yield "ostovar", code, sig
    if args.dataset in ("ceravolo", "both"):
        for code, sig in ddt.iter_ceravolo(args, channels):
            yield "ceravolo", code, sig


def summarize(rows: list[dict], label: str) -> dict:
    scored = len(rows)
    cu = sum(1 for r in rows if r["category"] == "correct_unambiguous")
    cr = sum(1 for r in rows if r["category"] == "correct_resolved")
    inc = sum(1 for r in rows if r["category"] == "incorrect")
    uns = sum(1 for r in rows if r["category"] == "unsupported")
    exact_acc = (cu + cr) / scored if scored else 0.0
    denom = scored - uns
    amb_acc = (cu + cr) / denom if denom else 0.0
    return {"dataset": label, "scored": scored, "correct_unambiguous": cu, "correct_resolved": cr,
            "incorrect": inc, "unsupported": uns, "exact_accuracy": round(exact_acc, 4),
            "ambiguity_aware_accuracy": round(amb_acc, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out-rows", type=Path, default=Path("results/label_accuracy_results.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("results/label_accuracy_summary.csv"))
    args = ap.parse_args()

    if args.sizes:
        unknown = {s.strip() for s in args.sizes.split(",") if s.strip()} - set(ddt.SIZES)
        if unknown:
            ap.error(f"unknown --sizes {sorted(unknown)}; choose from {','.join(ddt.SIZES)}")
    channels = {c.strip().lower() for c in args.channels.split(",") if c.strip()}
    if not channels or not channels <= ddt.CHANNELS:
        ap.error(f"--channels must be a non-empty subset of {sorted(ddt.CHANNELS)}; got {sorted(channels)}")

    floor = 1.0 - args.born_k
    rows: list[dict] = []
    idx_counters: dict[tuple[str, str], int] = {}
    for ds, code, sig in iter_drifts(args, channels):
        if sig is None:
            continue
        idx = idx_counters.get((ds, code), 0)
        idx_counters[(ds, code)] = idx + 1
        predicted = ddt.classify_sig(sig, floor)
        valid = VALID_LABELS.get(code, set())
        ambiguous = len(valid) > 1
        if predicted in valid:
            category = "correct_resolved" if ambiguous else "correct_unambiguous"
        elif predicted == "OTHER" and not valid:
            category = "unsupported"
        else:
            category = "incorrect"
        rows.append({"dataset": ds, "drift_id": f"{ds}:{code}:{idx}", "pattern_code": code,
                     "predicted_label": predicted, "valid_labels": "|".join(sorted(valid)),
                     "ambiguous": ambiguous, "category": category})

    if not rows:
        print("no scored drifts -- check --logs-dir/--ceravolo-dir/--sizes/--dataset", file=sys.stderr)
        return 2

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "drift_id", "pattern_code", "predicted_label",
                                            "valid_labels", "ambiguous", "category"])
        w.writeheader()
        w.writerows(rows)

    summary_rows = [summarize([r for r in rows if r["dataset"] == ds], ds)
                     for ds in sorted({r["dataset"] for r in rows})]
    summary_rows.append(summarize(rows, "ALL"))

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "scored", "correct_unambiguous", "correct_resolved",
                                            "incorrect", "unsupported", "exact_accuracy", "ambiguity_aware_accuracy"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Label accuracy | dataset={args.dataset} sizes={args.sizes or 'all'} | K_born={args.born_k}\n")
    print(f"  {'dataset':10}{'scored':>7}{'correct_unamb':>15}{'correct_resolved':>18}"
          f"{'incorrect':>11}{'unsupported':>13}{'exact_acc':>11}{'amb_aware_acc':>15}")
    for s in summary_rows:
        print(f"  {s['dataset']:10}{s['scored']:>7}{s['correct_unambiguous']:>15}{s['correct_resolved']:>18}"
              f"{s['incorrect']:>11}{s['unsupported']:>13}{s['exact_accuracy']:>11.3f}{s['ambiguity_aware_accuracy']:>15.3f}")
    print(f"\nWrote {args.out_rows} and {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
