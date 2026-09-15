#!/usr/bin/env python3
"""Qualitative traceability examples (eval_specs.md Experiment D).

Dumps the full evidence chain -- regime profiles, change signature, which gates fire and
why, the cascade resolution, and the final label -- for three hand-picked drifts that
cover different cascade paths:

  rp (Substitute)        Occurrence -> Substitution
  cm (ConditionalMove)   Coordination -> Branch, with Reorder also firing (multi-gate)
  cb (Skip)              Coordination -> Skip

All three patterns exist in Ostovar's PATTERN_MAP, so this defaults to --dataset ostovar;
override --codes/--dataset to pick different examples. Also reports the traceability
coverage: labelled_with_evidence / total_labelled (every non-OTHER label carries an
explicit predicate + the relations that triggered it, so this should be close to 1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import diagnose_decision_tree as ddt

EXAMPLE_CODES = ["rp", "cm", "cb"]
EXAMPLE_NAME = {"rp": "Substitute", "cm": "ConditionalMove", "cb": "Skip"}


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", choices=["ostovar", "ceravolo"], default="ostovar")
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
    ap.add_argument("--sizes", default="1000")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--span", type=int, default=0)
    ap.add_argument("--min-int-frac", type=float, default=0.0)


def _rel_str(rel) -> str:
    return f"{rel[0]}({rel[1]},{rel[2]})"


def gate_report(sig, floor: float) -> dict:
    occ = ddt.occ_shape(sig, floor)
    branch = ddt.branch_tc(sig)
    skip = ddt.skip_sig(sig)
    reorder = ddt.reorder_sig(sig)
    freq = ddt.freq_sig(sig)
    return {
        "OCC": {"fires": bool(occ), "detail": ("+".join(sorted(occ)) if occ else "no born/gone/dup/loop crossing")},
        "BRANCH": {"fires": branch, "detail": "Together(a,b) crosses 0<->positive among survivors" if branch else "no Together crossing among survivors"},
        "SKIP": {"fires": skip, "detail": "Existence(x,1) down-but-positive + incident ChainResponse down" if skip else "no down-but-positive Existence+ChainResponse pair"},
        "REORDER": {"fires": reorder, "detail": "mutual Response inversion, co-occurrence preserved" if reorder else "no order inversion (or Together also crossed)"},
        "FREQUENCY": {"fires": freq, "detail": "Response routing redistributes with no structural crossing" if freq else "no pure routing redistribution"},
    }


def build_example(code: str, ds: str, args, channels, floor: float) -> dict | None:
    it = ddt.iter_ostovar(args, channels) if ds == "ostovar" else ddt.iter_ceravolo(args, channels)
    for i, (c, sig) in enumerate(it):
        if c != code or sig is None:
            continue
        gates = gate_report(sig, floor)
        fired = [g for g, v in gates.items() if v["fires"]]
        label = ddt.classify_sig(sig, floor)
        changed = sorted(sig.born_relations | sig.died_relations | sig.shifted_relations)
        regime = [
            {"relation": _rel_str(rel), "supp_before": round(ddt._supp(sig.before, rel), 4),
             "supp_after": round(ddt._supp(sig.after, rel), 4)}
            for rel in changed[:20]
        ]
        return {
            "example_id": f"{EXAMPLE_NAME.get(code, code).lower()}_{code}_{i:02d}",
            "dataset": ds.capitalize(),
            "pattern": f"{code} ({EXAMPLE_NAME.get(code, code)})",
            "regime_changes": regime,
            "regime_changes_truncated": len(changed) > 20,
            "signature": {
                "born_relations": [_rel_str(r) for r in sorted(sig.born_relations)],
                "died_relations": [_rel_str(r) for r in sorted(sig.died_relations)],
                "shifted_relations": [_rel_str(r) for r in sorted(sig.shifted_relations)][:20],
                "born_activities": sorted(sig.born_activities),
                "died_activities": sorted(sig.died_activities),
            },
            "gates": gates,
            "gates_firing": fired,
            "ambiguous": len(fired) > 1,
            "cascade_winner": ddt.route({"occ": gates["OCC"]["fires"], "branch": gates["BRANCH"]["fires"],
                                          "skip": gates["SKIP"]["fires"], "reorder": gates["REORDER"]["fires"],
                                          "freq": gates["FREQUENCY"]["fires"]}),
            "label": label,
        }
    return None


def print_example(ex: dict) -> None:
    print(f"--- {ex['example_id']} ---")
    print(f"dataset: {ex['dataset']}")
    print(f"pattern: {ex['pattern']}")
    print("signature:")
    print(f"    born_relations:   {ex['signature']['born_relations']}")
    print(f"    died_relations:   {ex['signature']['died_relations']}")
    print(f"    shifted_relations: {ex['signature']['shifted_relations']}"
          + ("  (+more)" if len(ex['signature']['shifted_relations']) >= 20 else ""))
    print(f"    born_activities:  {ex['signature']['born_activities']}")
    print(f"    died_activities:  {ex['signature']['died_activities']}")
    print("gates:")
    for g in ("OCC", "BRANCH", "SKIP", "REORDER", "FREQUENCY"):
        v = ex["gates"][g]
        print(f"    {g:10} {'fires' if v['fires'] else 'does not fire':14} -- {v['detail']}")
    print(f"cascade:")
    print(f"    gates firing: {ex['gates_firing']}  (ambiguous={ex['ambiguous']})")
    print(f"    winner (precedence OCC>BRANCH>SKIP>REORDER>FREQUENCY): {ex['cascade_winner']}")
    print(f"label: {ex['label']}")
    print()


def traceability_coverage(args, channels, floor: float) -> tuple[int, int]:
    """labelled_with_evidence / total_labelled across BOTH datasets (Experiment B pool)."""
    both_args = argparse.Namespace(**vars(args))
    total = with_evidence = 0
    for code, sig in ddt.iter_ostovar(both_args, channels):
        if sig is None:
            continue
        total += 1
        if ddt.classify_sig(sig, floor) != "OTHER":
            with_evidence += 1
    for code, sig in ddt.iter_ceravolo(both_args, channels):
        if sig is None:
            continue
        total += 1
        if ddt.classify_sig(sig, floor) != "OTHER":
            with_evidence += 1
    return with_evidence, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--codes", default=",".join(EXAMPLE_CODES), help="comma-separated pattern codes to pick one example each")
    ap.add_argument("--out", type=Path, default=Path("results/traceability_examples.json"))
    args = ap.parse_args()

    channels = {c.strip().lower() for c in args.channels.split(",") if c.strip()}
    if not channels or not channels <= ddt.CHANNELS:
        ap.error(f"--channels must be a non-empty subset of {sorted(ddt.CHANNELS)}; got {sorted(channels)}")
    floor = 1.0 - args.born_k

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    examples = []
    for code in codes:
        ex = build_example(code, args.dataset, args, channels, floor)
        if ex is None:
            print(f"[warn] no scored drift found for pattern {code!r} in {args.dataset}", file=sys.stderr)
            continue
        examples.append(ex)
        print_example(ex)

    with_ev, total = traceability_coverage(args, channels, floor)
    coverage = with_ev / total if total else 0.0
    print(f"Traceability coverage (labelled_with_evidence / total_labelled, both datasets, "
          f"sizes={args.sizes or 'all'}): {with_ev}/{total} = {coverage:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump({"examples": examples, "traceability_coverage": {"with_evidence": with_ev, "total": total,
                                                                     "coverage": round(coverage, 4)}}, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
