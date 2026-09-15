#!/usr/bin/env python3
"""Per-batch support + confidence series for ChainResponse and Response relations.

Step 1 of the incremental drift-detection plan
(chatgpt-findings/10-incremental-drift-detection-plan.md). Turns one event log
into a tidy per-batch series with TWO channels per relation, the object the
threshold-free detector (step 2) consumes.

Relations (per case):
  ChainResponse(a, b)  -- direct succession / directly-follows.
  Response(a, b)       -- weak order (WOR): a occurs before b somewhere.
Both include self-pairs (a, a): immediate repeat / eventual repeat.

Two channels per relation (both are proportions -> same significance test):
  support     -- fraction of CASES containing the relation (binary presence).
                 Denominator = cases in the batch. Detects birth/death/flip of
                 adjacencies.
  confidence  -- fulfilled activations / total activations =
                 sum(fulfillments) / sum(activations of a) over the batch.
                 Denominator = activations. Detects FREQUENCY redistribution
                 among relations that stay present (e.g. parallel<->sequential),
                 which support alone misses.

Activations of both ChainResponse(a,*) and Response(a,*) = occurrences of a, so a
single per-activity activation total feeds every relation triggered by a.

Batching: --batch-size B consecutive case-starts; each case attributed to its
start batch.

Output: tidy CSV, one row per relation per batch, plus a batch manifest CSV.
"""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from log_io import log_base_name, parse_traces

DEFAULT_LOG = Path("cdrift-evaluation/EvaluationLogs/Ceravolo/sudden_trace_noise0_1000_sw.xes.gz")

Relation = tuple[str, str, str]  # (template, activity_a, activity_b)

RELATION_FAMILIES = ("chainresponse", "response", "existence", "coexistence", "choice")
DEFAULT_RELATIONS = "chainresponse,response,existence,choice"


def trace_counts_fulfillments(
    trace: tuple[str, ...], relations: set[str]
) -> tuple[dict[str, int], dict[Relation, int]]:
    """Per-trace activity occurrence counts and per-relation fulfillment counts.

    ChainResponse(a,b) fulfillments = number of adjacencies a,b.
    Response(a,b) fulfillments = number of a-positions with some b after
        (for a==b: count(a)-1; for a!=b: a-positions strictly before last[b]).
    """
    pos: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(trace):
        pos[name].append(i)
    counts = {k: len(v) for k, v in pos.items()}
    last = {k: v[-1] for k, v in pos.items()}

    ff: dict[Relation, int] = {}
    if "chainresponse" in relations:
        for i in range(len(trace) - 1):
            rel = ("ChainResponse", trace[i], trace[i + 1])
            ff[rel] = ff.get(rel, 0) + 1
    if "response" in relations:
        for a, apos in pos.items():
            ca = counts[a]
            for b, lb in last.items():
                f = (ca - 1) if a == b else bisect_left(apos, lb)
                if f > 0:
                    ff[("Response", a, b)] = f
    return counts, ff


def build_series(
    traces: list[tuple[str, ...]],
    batch_size: int,
    relations: set[str],
) -> tuple[list[Relation], list[dict[str, object]], dict[Relation, list[int]], dict[Relation, list[int]], dict[Relation, list[int]]]:
    """Compute per-batch (support, fulfillments, activations) per relation.

    Families (selected via `relations`):
      chainresponse / response -- order, with support + confidence channels.
      existence                -- Existence(x, n) for every observed n: per-case
                                   binary count(x) >= n. Support-only (no confidence).
                                   Keyed ("Existence", x, str(n)).
      coexistence              -- CoExistence(a, b): per-case (a in t) == (b in t),
                                   once per unordered pair. Support-only.
                                   Keyed ("CoExistence", a, b) with a < b.
    Support-only families are omitted from `activations`/`fulfillments`, so the
    detector's confidence channel skips them automatically.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    unknown = relations - set(RELATION_FAMILIES)
    if unknown:
        raise ValueError(f"unknown relation families: {sorted(unknown)}")

    n_batches = (len(traces) + batch_size - 1) // batch_size
    activities = sorted({a for trace in traces for a in trace})
    need_pairs = relations & {"coexistence", "choice"}
    pairs = [(a, b) for i, a in enumerate(activities) for b in activities[i + 1:]] if need_pairs else []

    support: dict[Relation, list[int]] = {}
    fulfillments: dict[Relation, list[int]] = {}
    act_by_activity: dict[str, list[int]] = {}
    manifest: list[dict[str, object]] = []

    for batch_index in range(n_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(traces))
        batch = traces[start:end]
        event_count = 0
        for trace in batch:
            event_count += len(trace)
            counts, ff = trace_counts_fulfillments(trace, relations)
            for a, ca in counts.items():
                act_by_activity.setdefault(a, [0] * n_batches)[batch_index] += ca
            for rel, f in ff.items():
                fulfillments.setdefault(rel, [0] * n_batches)[batch_index] += f
                # one case, one vote -- ff only holds positive counts
                support.setdefault(rel, [0] * n_batches)[batch_index] += 1
            if "existence" in relations:
                for a, ca in counts.items():
                    for nlevel in range(1, ca + 1):
                        support.setdefault(("Existence", a, str(nlevel)), [0] * n_batches)[batch_index] += 1
            if need_pairs:
                present = counts.keys()
                for a, b in pairs:
                    if "coexistence" in relations and (a in present) == (b in present):
                        support.setdefault(("CoExistence", a, b), [0] * n_batches)[batch_index] += 1
                    if "choice" in relations and (a in present or b in present):
                        support.setdefault(("Choice", a, b), [0] * n_batches)[batch_index] += 1
        manifest.append(
            {
                "batch_index": batch_index,
                "start_case": start,
                "end_case": end - 1,
                "n_cases": len(batch),
                "event_count": event_count,
            }
        )

    # Derive co-occurrence "Together(a,b)" = # cases with BOTH a and b, EXACTLY from
    # standard Declare template counts: #both = #a + #b - #(a OR b)
    #   = Existence(a,1) + Existence(b,1) - Choice(a,b). No bespoke template.
    if "choice" in relations and "existence" in relations:
        for rel in [r for r in support if r[0] == "Choice"]:
            _, a, b = rel
            ea = support.get(("Existence", a, "1"))
            eb = support.get(("Existence", b, "1"))
            if ea is not None and eb is not None:
                ch = support[rel]
                tog = [max(0, ea[t] + eb[t] - ch[t]) for t in range(n_batches)]
                support[("Together", a, b)] = tog
                # ExclusiveChoice(a,b) = exactly-one = Choice - Together (standard Declare,
                # excludes the "neither" case -> sparse exclusivity signal for conditional branches).
                support[("ExclusiveChoice", a, b)] = [max(0, ch[t] - tog[t]) for t in range(n_batches)]

    relation_list = sorted(set(support) | set(fulfillments))
    # Activations only for order families -> support-only families skip the confidence channel.
    activations = {rel: list(act_by_activity[rel[1]]) for rel in relation_list if rel[0] in ("ChainResponse", "Response") and rel[1] in act_by_activity}
    return relation_list, manifest, support, fulfillments, activations


def write_series_csv(
    relation_list: list[Relation],
    manifest: list[dict[str, object]],
    support: dict[Relation, list[int]],
    fulfillments: dict[Relation, list[int]],
    activations: dict[Relation, list[int]],
    output: Path,
) -> None:
    assert set(fulfillments) <= set(support)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "template",
                "activity_a",
                "activity_b",
                "batch_index",
                "start_case",
                "end_case",
                "n_cases",
                "support",
                "support_pct",
                "activations",
                "fulfillments",
                "confidence_pct",
            ],
        )
        writer.writeheader()
        for rel in relation_list:
            template, a, b = rel
            for batch in manifest:
                bi = int(batch["batch_index"])
                n_cases = int(batch["n_cases"])
                sup = support[rel][bi]
                act = activations[rel][bi] if rel in activations else 0
                ff = fulfillments[rel][bi] if rel in fulfillments else 0
                writer.writerow(
                    {
                        "template": template,
                        "activity_a": a,
                        "activity_b": b,
                        "batch_index": bi,
                        "start_case": batch["start_case"],
                        "end_case": batch["end_case"],
                        "n_cases": n_cases,
                        "support": sup,
                        "support_pct": f"{(100.0 * sup / n_cases) if n_cases else 0.0:.4f}",
                        "activations": act,
                        "fulfillments": ff,
                        "confidence_pct": f"{(100.0 * ff / act):.4f}" if act else "",
                    }
                )


def write_manifest_csv(manifest: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["batch_index", "start_case", "end_case", "n_cases", "event_count"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="XES or XES.GZ event log.")
    parser.add_argument("--activity-key", default="concept:name", help="Event attribute used as the activity name.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of case-starts per batch (B).")
    parser.add_argument(
        "--relations",
        default=DEFAULT_RELATIONS,
        help=f"Comma-separated subset of {{{','.join(RELATION_FAMILIES)}}}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("declare-template-results"))
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    args = parser.parse_args()

    relations = {r.strip().lower() for r in args.relations.split(",") if r.strip()}
    unknown = relations - set(RELATION_FAMILIES)
    if unknown:
        parser.error(f"unknown relations: {sorted(unknown)}; choose from {','.join(RELATION_FAMILIES)}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    traces = parse_traces(args.log, args.activity_key)
    if not traces:
        parser.error(f"no traces parsed from {args.log}")

    relation_list, manifest, support, fulfillments, activations = build_series(traces, args.batch_size, relations)

    base = log_base_name(args.log)
    csv_path = args.csv_output or args.output_dir / f"{base}-support-series-B{args.batch_size}.csv"
    manifest_path = args.manifest_output or args.output_dir / f"{base}-support-series-B{args.batch_size}-batches.csv"

    write_series_csv(relation_list, manifest, support, fulfillments, activations, csv_path)
    write_manifest_csv(manifest, manifest_path)

    print(f"Parsed {len(traces)} cases from {args.log}")
    print(f"Batches: {len(manifest)} (B={args.batch_size}); relations tracked: {len(relation_list)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
