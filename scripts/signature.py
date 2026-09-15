"""Change-point evidence and the typed change signature for labeling (Phase B).

No magnitude thresholds here -- "moved enough?" is decided by the detector's
significance test. Presence facts are structural (0 vs positive), not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from detect_drift_incremental import Series, Shift, classify, critical_value, significant

Relation = tuple[str, str, str]  # (template, activity_a, activity_b)


@dataclass(frozen=True)
class RegimeProfile:
    """Aggregate of one regime (a contiguous run of batches)."""

    relation_support: dict[Relation, float]       # fraction of cases with the relation
    activities: set[str]                          # activities occurring at all
    n_cases: int


@dataclass(frozen=True)
class ChangePointEvidence:
    case_index: int
    before: RegimeProfile
    after: RegimeProfile
    shifts: list[Shift]


@dataclass(frozen=True)
class ChangeSignature:
    """Structural classification of a change point, the input to signal extractors."""

    born_relations: set[Relation] = field(default_factory=set)     # appeared (presence 0 -> positive)
    died_relations: set[Relation] = field(default_factory=set)     # disappeared
    shifted_relations: set[Relation] = field(default_factory=set)  # rate moved, still present
    born_activities: set[str] = field(default_factory=set)
    died_activities: set[str] = field(default_factory=set)
    present_both: set[str] = field(default_factory=set)
    before: RegimeProfile | None = None
    after: RegimeProfile | None = None


def _profile(series: Series, lo: int, hi: int) -> RegimeProfile:
    """Aggregate batches [lo, hi) into a RegimeProfile."""
    n_cases = [b.n_cases for b in series.batches]
    total = sum(n_cases[lo:hi])
    support: dict[Relation, float] = {}
    activities: set[str] = set()
    relations = set(series.support) | set(series.fulfillments)
    for rel in relations:
        s = sum(series.support.get(rel, [])[lo:hi])
        if s > 0:
            support[rel] = s / total
            # Derive "activity occurs" ONLY from relations that evidence occurrence.
            # CoExistence is satisfied by the "neither" case too, so it must NOT imply
            # the activity occurred (that pollutes born/died detection).
            if rel[0] in ("ChainResponse", "Response"):
                activities.add(rel[1])
                activities.add(rel[2])
            elif rel[0] == "Existence":
                activities.add(rel[1])  # activity_b holds the cardinality level n, not an activity
    return RegimeProfile(support, activities, total)


def build_evidence(series: Series, b_lo: int, b_hi: int, a_lo: int, a_hi: int, channels: set[str], test: str, confidence: float) -> ChangePointEvidence:
    """Evidence comparing a clean before-regime [b_lo, b_hi) vs after-regime [a_lo, a_hi).

    The two ranges may be non-adjacent (b_hi <= a_lo): transition batches between a
    drift's onset-spread change points are skipped, so born/died classification is
    not contaminated by the partially-drifted middle.
    """
    before = _profile(series, b_lo, b_hi)
    after = _profile(series, a_lo, a_hi)
    n_cases = [b.n_cases for b in series.batches]
    crit = critical_value(test, confidence)
    shifts: list[Shift] = []
    for rel in sorted(set(series.support) | set(series.fulfillments)):
        if "support" in channels:
            s = series.support.get(rel)
            if s is not None:
                x1, n1 = sum(s[b_lo:b_hi]), sum(n_cases[b_lo:b_hi])
                x2, n2 = sum(s[a_lo:a_hi]), sum(n_cases[a_lo:a_hi])
                sig, stat = significant(x1, n1, x2, n2, test, crit)
                if sig:
                    shifts.append(Shift(rel, "support", classify(x1, x2), stat))
        if "confidence" in channels:
            f, a = series.fulfillments.get(rel), series.activations.get(rel)
            if f is not None and a is not None:
                x1, n1 = sum(f[b_lo:b_hi]), sum(a[b_lo:b_hi])
                x2, n2 = sum(f[a_lo:a_hi]), sum(a[a_lo:a_hi])
                sig, stat = significant(x1, n1, x2, n2, test, crit)
                if sig:
                    # Forced "shift", never classify(): born/died is a support (presence)
                    # concept. See classify()'s docstring -- a confidence move is a rate
                    # change on an already-present relation, never a 0<->positive crossing.
                    shifts.append(Shift(rel, "confidence", "shift", stat))
    return ChangePointEvidence(series.batches[a_lo].start_case, before, after, shifts)


def build_change_signature(ev: ChangePointEvidence) -> ChangeSignature:
    """Classify the evidence into born/died/shifted relations + born/died activities.

    A relation is born if it has a 'new' shift on any channel, died if 'disappeared',
    else shifted. Activity birth/death is the structural presence change between the
    two regimes (occurs at all: 0 <-> positive)."""
    kinds: dict[Relation, set[str]] = {}
    for sh in ev.shifts:
        kinds.setdefault(sh.relation, set()).add(sh.kind)
    born = {r for r, ks in kinds.items() if "new" in ks}
    died = {r for r, ks in kinds.items() if "disappeared" in ks and "new" not in ks}
    shifted = {r for r, ks in kinds.items() if ks == {"shift"}}
    return ChangeSignature(
        born_relations=born,
        died_relations=died,
        shifted_relations=shifted,
        born_activities=ev.after.activities - ev.before.activities,
        died_activities=ev.before.activities - ev.after.activities,
        present_both=ev.before.activities & ev.after.activities,
        before=ev.before,
        after=ev.after,
    )
