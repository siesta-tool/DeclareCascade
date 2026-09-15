#!/usr/bin/env python3
"""Threshold-free drift detection from a per-batch support+confidence series.

Consumes the two-channel per-batch series built by batch_support_series.py.
There is no magnitude gate -- the only control is a statistical confidence K.

Channels (both are proportions, so both take the same test):
  support     -- fraction of cases with the relation (denominator = cases).
                 Detects birth/death/flip of adjacencies.
  confidence  -- fulfilled activations / activations (denominator = activations).
                 Detects frequency redistribution among relations that stay
                 present (parallel<->sequential, synchronize), which support misses.

Significance, before=(x1 of n1) vs after=(x2 of n2), pooled p,
SE=sqrt(p(1-p)(1/n1+1/n2)):
  z (default)  flag if |p1-p2|/SE > z*(K).
  chebyshev    flag if |p1-p2|/SE > 1/sqrt(1-K)  (distribution-free).
  fisher       exact two-sided test on the 2x2 table (requires scipy).

sequential_detect scans boundaries left to right against a baseline that grows
from the current anchor, and re-anchors past every firing -- so a log with
several drifts yields several change points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

try:
    from scipy.stats import fisher_exact
except ImportError:  # only test="fisher" needs it
    fisher_exact = None

Relation = tuple[str, str, str]


@dataclass(frozen=True)
class Batch:
    index: int
    start_case: int
    end_case: int
    n_cases: int


@dataclass(frozen=True)
class Shift:
    relation: Relation
    channel: str  # "support" | "confidence"
    kind: str  # "new" | "disappeared" | "shift"
    stat: float  # z-magnitude; kept for diagnostics


@dataclass(frozen=True)
class Series:
    batches: list[Batch]
    support: dict[Relation, list[int]]
    fulfillments: dict[Relation, list[int]]
    activations: dict[Relation, list[int]]


@dataclass(frozen=True)
class ChangePoint:
    boundary: int
    case_index: int
    intensity: int  # significant-relation count at the split


# --------------------------------------------------------------------------- #
# Significance.
# --------------------------------------------------------------------------- #

def critical_value(test: str, confidence: float) -> float:
    if test == "z":
        return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    if test == "chebyshev":
        return 1.0 / math.sqrt(1.0 - confidence)
    if test == "fisher":
        return 1.0 - confidence  # significance level alpha; decision uses an exact p-value
    raise ValueError(f"unknown test: {test}")


def shift_stat(x1: int, n1: int, x2: int, n2: int) -> float:
    """Standardised two-proportion difference |p1-p2| / SE (0 if undefined).

    Always reported as a magnitude indicator; the significance DECISION uses the
    chosen test (which, for fisher, is an exact p-value rather than this z-stat).
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0
    return abs(p1 - p2) / se


def significant(x1: int, n1: int, x2: int, n2: int, test: str, crit: float) -> tuple[bool, float]:
    """(is the support/confidence shift significant, z-magnitude for display).

    z / chebyshev compare a normal-approx z-statistic to a critical value (invalid
    when n*p is tiny). fisher: exact two-sided test on the 2x2 table -- correct for
    small counts, where the normal approximation is not. Every published result uses
    z; fisher is available but unvalidated on the benchmark.
    """
    stat = shift_stat(x1, n1, x2, n2)
    if n1 == 0 or n2 == 0:  # shift_stat already returned 0.0; this guard controls the DECISION.
        return False, stat
    if test == "fisher":
        if fisher_exact is None:
            raise SystemExit("--test fisher requires scipy (pip install scipy)")
        _odds, p = fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        return p < crit, stat
    return stat > crit, stat


def classify(x1: int, x2: int) -> str:
    """new / disappeared / shift from before- and after-counts.

    Callers on the CONFIDENCE channel must force "shift" rather than use this result:
    born/died is a SUPPORT (presence) concept, and a confidence move is always a rate
    change on an already-present relation, never a 0<->positive crossing.
    """
    if x1 == 0 and x2 > 0:
        return "new"
    if x1 > 0 and x2 == 0:
        return "disappeared"
    return "shift"


# --------------------------------------------------------------------------- #
# Detection.
# --------------------------------------------------------------------------- #

def sequential_detect(
    series: Series, channels: set[str], test: str, confidence: float, span: int, templates: set[str] | None = None
) -> list[ChangePoint]:
    """Sequential scan with forgetting (LCDD-style).

    The baseline grows from the current anchor; fires when BOTH channels flag a
    change, then re-anchors past it. Controls: K (significance) and span
    (after-window / persistence). Recall-oriented -- false positives are left to
    the labeling phase.

    Two caveats, both measured:
      * Only boundaries in [span, n-span] are testable; with span=2 on a 5-batch log
        that is 2 of 4 real boundaries, and a fire lands one batch early.
      * The two channels are correlated (support is derived from the fulfillment
        counts), so the agreement gate is far weaker than K^2. On a permutation null
        over real Ceravolo logs the confidence channel's realized alpha is ~15x
        nominal (0.0146 vs 0.001) because activations are clustered within cases --
        so K bounds the support channel, not the pair.
    """
    batches = series.batches
    n = len(batches)
    n_cases = [b.n_cases for b in batches]
    crit = critical_value(test, confidence)
    relations = sorted(set(series.support) | set(series.fulfillments))
    if templates is not None:  # restrict firing to chosen families (e.g. order only)
        relations = [r for r in relations if r[0] in templates]

    results: list[ChangePoint] = []
    anchor = 0
    t = anchor + span  # baseline [anchor, t-1] holds >= span batches; after = [t, t+span-1]
    while t + span <= n:
        sup_rels, con_rels = set(), set()
        for rel in relations:
            if "support" in channels:
                s = series.support.get(rel)
                if s is not None:
                    sig, _ = significant(sum(s[anchor:t]), sum(n_cases[anchor:t]), sum(s[t:t + span]), sum(n_cases[t:t + span]), test, crit)
                    if sig:
                        sup_rels.add(rel)
            if "confidence" in channels:
                f, a = series.fulfillments.get(rel), series.activations.get(rel)
                if f is not None and a is not None:
                    sig, _ = significant(sum(f[anchor:t]), sum(a[anchor:t]), sum(f[t:t + span]), sum(a[t:t + span]), test, crit)
                    if sig:
                        con_rels.add(rel)
        fires = (sup_rels and con_rels) if ("support" in channels and "confidence" in channels) \
            else bool(sup_rels or con_rels)  # single-channel ablation: that one channel alone gates
        if fires:
            results.append(ChangePoint(t, batches[t].start_case, len(sup_rels | con_rels)))
            anchor = t
            t = anchor + span
        else:
            t += 1
    return results


def auto_config(n_cases: int) -> tuple[int, int]:
    """Heuristic (batch_size, span) from log length.

    Target ~100 cases per batch for stable proportion estimates where the log allows
    it; for short logs fall back to ~10 batches. Use span=2 when that yields few
    batches (<=6), else span=1.

    Derived from the Ceravolo cross-size sweep. Note the cliff: 499 cases -> (50, span 1,
    10 batches); 500 -> (100, span 2, 5 batches). One extra trace halves the batch count
    and doubles the span, and span=2 leaves only boundaries [2, n-2] testable.
    """
    batch_size = 100 if n_cases >= 500 else max(2, round(n_cases / 10))
    n_batches = (n_cases + batch_size - 1) // batch_size
    span = 2 if n_batches <= 6 else 1
    return batch_size, span


def resolve_config(n_cases: int, batch_size: int | None = None,
                   span: int | None = None) -> tuple[int, int]:
    """(batch_size, span) with explicit overrides layered on top of auto_config(n_cases).

    Precedence (shared by every entry point -- cdrift_approach.detect_changepoints and
    diagnose_decision_tree._auto_cfg both delegate here):
      * batch_size given -> use it AND re-derive span from the NEW batch count, unless span is
        given too. Overriding B alone changes n_batches, and the span rule is a function of
        n_batches; carrying the auto span over would pair a hand-picked B with a span chosen
        for a different batch count.
      * span given alone -> keep the auto batch size, override span only.
      * neither          -> auto_config unchanged.

    Both 0 and None mean "not given" (0 is the CLI sentinel already used at
    diagnose_decision_tree.py's --batch-size/--span, so a flag forwards straight through; None is
    the natural library default). Negatives are a caller bug, not a sentinel, and raise.

    Note span=0 is silently treated as "auto" rather than erroring, even though a genuine span=0
    would make sequential_detect's anchor==t baseline sum to zero and its n1==0 guard return
    False forever -- i.e. a silent empty result, not a crash. That degenerate case is reachable
    only by explicitly requesting span=0, which this function defines to mean "auto" instead.
    """
    if batch_size is not None and batch_size < 0:
        raise ValueError(f"batch_size must be >= 0 (0 = auto), got {batch_size}")
    if span is not None and span < 0:
        raise ValueError(f"span must be >= 0 (0 = auto), got {span}")
    B, s = auto_config(n_cases)
    if batch_size:
        B = batch_size
        n_batches = (n_cases + B - 1) // B
        s = span if span else (2 if n_batches <= 6 else 1)
    elif span:
        s = span
    return B, s
