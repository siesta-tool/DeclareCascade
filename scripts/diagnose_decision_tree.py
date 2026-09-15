#!/usr/bin/env python3
"""Label a detected concept drift by routing its change signature down a five-gate tree.

Gates, in precedence order -- the first that fires wins:

  OCC        an activity's occurrence count changed   Existence(x,n)
  BRANCH     two present activities start/stop co-occurring   Together(a,b) crossing
  SKIP       an activity stays but participates less   Existence(x,1) down + ChainResponse down
  REORDER    a pair's eventual order inverts, co-occurrence preserved   Response inversion
  FREQUENCY  routing redistributes, nothing structural crosses   Response shifts only

OCC then sub-splits into INSERTION / REMOVAL / SUBSTITUTION / DUPLICATION / LOOP /
COMPOSITE. Each gate is one threshold-free Declare lens: every "changed" fact is either a
significance-tested shift the detector already found or a 0-vs-positive presence flip, so
the only tunable is the confidence K.

`--level` picks the report: LEAVES / SUMMARY for the tree, DETECT / INTENSITY / CONFIRM for
the detector, and the per-gate probes (OCC, BRANCH, ...) for gate purity.
`overview`, `L1` and `COND` are a legacy L1..L4 separability probe kept for continuity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from batch_support_series import build_series
from detect_drift_incremental import Batch, Series, resolve_config, sequential_detect
from log_io import log_base_name, parse_traces
from signature import build_change_signature, build_evidence

# ---- benchmark ground-truth (inlined; this script is self-contained) ----
# Ostovar: two drifts per log at fixed case indices; pattern in the file name.
NAME_RE = re.compile(r"(?:Atomic|Composite)_([A-Za-z]+)_output_")
TRUE_CPS = [999, 1999]
# Ostovar change-pattern name -> the short code used everywhere below. Ostovar's own
# change-family tags (insertion / optionalization / resequentialization / substitution /
# loop / frequency / branch) are benchmark metadata the tree never reads.
PATTERN_MAP = {
    "Swap": "sw", "Skip": "cb", "Substitute": "rp", "SerialRemoval": "re",
    "SerialMove": "sm", "Loop": "lp", "Frequency": "fr",
    "ConditionalToSequence": "cf", "ConditionalMove": "cm", "ConditionalRemoval": "cre",
    "ParallelToSequence": "pl", "ParallelMove": "pm", "ParallelRemoval": "pre",
    "IOR": "ior", "IRO": "iro", "OIR": "oir", "ORI": "ori", "RIO": "rio", "ROI": "roi",
}
# Ceravolo: one drift per log; size + pattern in the name; true changepoint from the CSV.
CER_NAME_RE = re.compile(r"noise\d+_(\d+)_(.+)$")
SIZES = ["100", "500", "1000"]
CHANNELS = {"support", "confidence"}          # the two series the detector can test


def events_of(cps):
    """Cluster adjacent change-point boundaries into onset-spread events (first,last batch)."""
    bs = sorted(cp.boundary for cp in cps)
    events = []
    if bs:
        start = prev = bs[0]
        for b in bs[1:]:
            if b - prev <= 1:
                prev = b
            else:
                events.append((start, prev)); start = prev = b
        events.append((start, prev))
    return events


def load_ground_truth(results_csv: Path) -> dict[str, int]:
    """Ceravolo true first-changepoint per log, from the cdrift results CSV."""
    gt: dict[str, int] = {}
    with results_csv.open() as fh:
        for row in csv.DictReader(fh):
            if (row.get("Log Source") or "") != "Ceravolo":
                continue
            log = row["Log"]
            if log not in gt:
                cps = ast.literal_eval(row["Actual Changepoints for Log"])
                if cps:
                    gt[log] = int(cps[0])
    return gt

# Conceptually-correct top-level bucket(s) for each Ostovar pattern. A pattern may
# legitimately span buckets (composites = insertion + reorder [+ optionalization]).
EXPECTED_LEVEL = {
    "re": {"L1"}, "rp": {"L1"},
    "lp": {"L2"},
    "cb": {"L3"},
    "sw": {"L4"}, "sm": {"L4"}, "fr": {"L4"},
    "cf": {"L4"}, "cm": {"L4"}, "pl": {"L4"}, "pm": {"L4"},
    "cre": {"L1", "L4"}, "pre": {"L1", "L4"},
    "ior": {"L1", "L4"}, "iro": {"L1", "L4"}, "oir": {"L1", "L4"},
    "ori": {"L1", "L4"}, "rio": {"L1", "L4"}, "roi": {"L1", "L4"},
}

ORDER_TEMPLATES = {"ChainResponse", "Response", "ExclusiveChoice", "Together", "Choice"}


def _supp(prof, rel) -> float:
    return prof.relation_support.get(rel, 0.0) if prof else 0.0


def _by_code(rows: list[dict]) -> dict[str, list[dict]]:
    """Group scored/missed drift rows by benchmark pattern code."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["code"], []).append(r)
    return by


def decisions(sig) -> set[str]:
    """Which of the four top-level decisions fire for this change signature."""
    changed = sig.born_relations | sig.died_relations | sig.shifted_relations
    fired: set[str] = set()

    # L1 -- alphabet membership changed (an activity appears or disappears at all).
    if sig.born_activities or sig.died_activities:
        fired.add("L1")

    # L2 -- a surviving activity starts repeating: Existence(x, n>=2) support up.
    card_up: set[str] = set()
    for rel in changed:
        if (rel[0] == "Existence" and rel[2].isdigit() and int(rel[2]) >= 2
                and _supp(sig.after, rel) > _supp(sig.before, rel)):
            card_up.add(rel[1])
    if card_up:
        fired.add("L2")

    # L3 -- a surviving activity's presence-rate shifts (Existence(x,1)) without card-up.
    mand: set[str] = set()
    for rel in changed:
        if rel[0] == "Existence" and rel[2] == "1" and rel[1] in sig.present_both and rel[1] not in card_up:
            mand.add(rel[1])
    if mand:
        fired.add("L3")

    # L4 -- the relations among SURVIVING activities change (type/order/routing).
    for rel in changed:
        if rel[0] in ORDER_TEMPLATES and rel[1] != rel[2] \
                and rel[1] in sig.present_both and rel[2] in sig.present_both:
            fired.add("L4")
            break

    return fired


def fingerprint(fired: set[str]) -> str:
    return "{" + ",".join(sorted(fired)) + "}" if fired else "{none}"


def l4_sublens(sig) -> set[str]:
    """Within L4 (relations among survivors changed), which Declare sub-lens fired?

    EC  -- ExclusiveChoice(a,b) crossed 0<->positive (a genuine alternative toggle):
           the choice/conditional discriminator. Requires BOTH endpoints to change
           cardinality so a constant partner (e.g. DRIFT_POINT) can't fake it.
    ORD -- an order relation (Response/ChainResponse) among survivors flipped/shifted.
    TOG -- Together(a,b) (co-occurrence) shifted, ExclusiveChoice did not.
    """
    sub: set[str] = set()
    born_died = sig.born_relations | sig.died_relations
    changed = born_died | sig.shifted_relations
    # activities whose own cardinality/presence shifted (Existence(x,*) changed)
    exist_changed = {rel[1] for rel in changed if rel[0] == "Existence"}
    for rel in changed:
        if rel[1] == rel[2] or rel[1] not in sig.present_both or rel[2] not in sig.present_both:
            continue
        t = rel[0]
        if t == "ExclusiveChoice" and rel in born_died:
            sub.add("EC")
            # ECg = both-endpoints-changed gate: a genuine alternative toggles BOTH
            # branches; a skip (cb) makes only ONE activity optional (one-sided exclusive).
            if rel[1] in exist_changed and rel[2] in exist_changed:
                sub.add("ECg")
        elif t in ("Response", "ChainResponse"):
            sub.add("ORD")
        elif t == "Together":
            sub.add("TOG")
    return sub


# Which patterns SHOULD fire each level (ground-truth membership for gate-purity). The same
# set is used for both datasets so the gate is judged on identical criteria -- Ceravolo's `re`
# is dup-like rather than a removal, so it shows up as a recall difference.
L1_POSITIVE = {"re", "rp", "cre", "pre", "ior", "iro", "oir", "ori", "rio", "roi"}
# Within-L1 expected born/died shape. cre/pre also carry a branch edit, but l1_shape() only
# reports the alphabet half, so the expectation here is the alphabet shape alone.
L1_SHAPE = {"re": "died-only", "rp": "born+died", "cre": "died-only", "pre": "died-only"}


def l1_shape(sig) -> str:
    b, d = bool(sig.born_activities), bool(sig.died_activities)
    if b and d:
        return "born+died"
    if b:
        return "born-only"
    if d:
        return "died-only"
    return "none"


# ---- OCC: the merged occurrence branch (insertion / removal / duplication / loop) ----
# One branch for "an activity's occurrence count changed", split internally by whether
# the activity was ALREADY present (dataset-agnostic; no per-benchmark membership).
OCC_POSITIVE = {"re", "rp", "cp", "lp", "cre", "pre", "ior", "iro", "oir", "ori", "rio", "roi"}
OCC_EXPECT = {                       # expected within-OCC shape(s)
    "re": "NEW|GONE (insert/remove direction)",
    "rp": "NEW+GONE (substitution)",
    "cp": "DUP",
    "lp": "LOOP",
    "cre": "GONE (+branch)",
    "pre": "GONE (+branch)",
}


def occ_born_gone(sig, floor: float):
    """Activities crossing the born-floor: (newly-present, newly-absent)."""
    changed = sig.born_relations | sig.died_relations | sig.shifted_relations
    sig_ex1 = {r[1] for r in changed if r[0] == "Existence" and r[2] == "1"}
    new_acts, gone_acts = set(), set()
    for x in sig_ex1:
        b = _supp(sig.before, ("Existence", x, "1"))
        a = _supp(sig.after, ("Existence", x, "1"))
        if b < floor <= a:
            new_acts.add(x)
        elif a < floor <= b:
            gone_acts.add(x)
    return new_acts, gone_acts


def occ_shape(sig, born_floor: float) -> set[str]:
    """Within the occurrence branch: which sub-operations the signature shows.

    NEW  -- Existence(x,1) crosses the born-floor upwards (insertion, including diluted ones).
    GONE -- Existence(x,1) crosses it downwards (removal).
    DUP  -- Existence(x,2) rises: a duplicate adds exactly one occurrence, so only level 2 crosses.
    LOOP -- Existence(x,3+) also rises: iteration produces a geometric cascade of levels.

    Why a born-floor (= 1 - K_born) rather than an exact 0-crossing: one contaminating trace
    (Existence(x,1)=0.001 before) would otherwise block a genuine insertion. The two real
    clusters are far apart -- diluted insertions <=0.7%, mandatoriness saturation >=30% -- so
    any floor in (0.007, 0.3) separates them. Measured sweep knee: K_born=0.99 (floor 0.01)
    gives OCC precision 0.98 / recall 0.89 on Ostovar; 0.95 drops precision to 0.83, and 0.995
    falls below the diluted 0.7% and loses recall.
    """
    out: set[str] = set()
    changed = sig.born_relations | sig.died_relations | sig.shifted_relations
    new_acts, gone_acts = occ_born_gone(sig, born_floor)
    if new_acts:
        out.add("NEW")
    if gone_acts:
        out.add("GONE")
    # LOOP iff the activity reaches Existence level >=3. The immediate ChainResponse(x,x)
    # signal is useless here: these loops cycle through other activities, so it never fires.
    card_levels: dict[str, set] = {}
    for rel in changed:
        if rel[0] == "Existence" and rel[2].isdigit() and int(rel[2]) >= 2 \
                and rel[1] in sig.present_both and _supp(sig.after, rel) > _supp(sig.before, rel):
            card_levels.setdefault(rel[1], set()).add(int(rel[2]))
    for levels in card_levels.values():
        out.add("LOOP" if max(levels) >= 3 else "DUP")
    return out


def report_occ(rows: list[dict]) -> None:
    """Gate purity for the occurrence branch, plus the within-OCC shape breakdown."""
    by = _by_code(rows)

    print("OCC (OCCURRENCE: insertion / removal / duplication / loop) — gate purity\n")
    print(f"{'pat':5}{'OCC-fires':>11}{'scored':>8}  membership   within-OCC shapes")
    fp = fn = tp = tn = 0
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        fires = sum(1 for r in rs if r.get("occ"))
        pos = code in OCC_POSITIVE
        shapes = Counter(r["occfp"] for r in rs if r.get("occ"))
        shp = ", ".join(f"{s}:{c}" for s, c in shapes.most_common()) or "-"
        if pos:
            tp += fires; fn += (scored - fires)
        else:
            fp += fires; tn += (scored - fires)
        print(f"{code:5}{f'{fires}/{scored}':>11}{scored:>8}  {'POS' if pos else 'neg':10}   {shp}")
    print(f"\nGate: occurrence patterns detected (recall) {tp}/{tp+fn}; "
          f"non-occurrence correctly silent (specificity) {tn}/{tn+fp}; false-fires {fp}")
    print("\nWithin-OCC separation (does shape pick out the sub-operation?):")
    for code in sorted(OCC_EXPECT):
        if code in by:
            shapes = Counter(r["occfp"] for r in by[code] if r.get("occ"))
            got = ", ".join(f"{s}:{c}" for s, c in shapes.most_common()) or "(never fires OCC)"
            print(f"  {code:5} expect {OCC_EXPECT[code]:32} got {got}")


def report_l1(rows: list[dict]) -> None:
    """Gate purity for the ALPHABET level, plus the born/died shape per pattern
    (died-only=removal, born-only=insertion, born+died=substitution)."""
    by = _by_code(rows)

    print("L1 (ALPHABET) — gate purity\n")
    print(f"{'pat':5}{'L1-fires':>10}{'scored':>8}  membership   within-L1 born/died shapes")
    fp = fn = tp = tn = 0
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        fires = sum(1 for r in rs if r.get("L1"))
        pos = code in L1_POSITIVE
        shapes = Counter(r["shape"] for r in rs if r.get("L1"))
        shp = ", ".join(f"{s}:{c}" for s, c in shapes.most_common()) or "-"
        if pos:
            tp += fires; fn += (scored - fires)
        else:
            fp += fires; tn += (scored - fires)
        mark = "POS" if pos else "neg"
        print(f"{code:5}{f'{fires}/{scored}':>10}{scored:>8}  {mark:10}   {shp}")
    print(f"\nGate: positives detected (recall) {tp}/{tp+fn}; "
          f"negatives correctly silent (specificity) {tn}/{tn+fp}; false-fires on negatives {fp}")
    print("\nWithin-L1 separation (expected shape per pattern):")
    for code in sorted(L1_SHAPE):
        if code in by:
            shapes = Counter(r["shape"] for r in by[code] if r.get("L1"))
            got = ", ".join(f"{s}:{c}" for s, c in shapes.most_common()) or "(never fires L1)"
            print(f"  {code:5} expect {L1_SHAPE[code]:16} got {got}")


# ---- REORDER bucket: swap + serial-move + parallel-move (one cluster) ----
# Swap and move are near-degenerate in standard Declare (both reorder while preserving
# co-occurrence), so they form ONE bucket. Signal: a mutual Response inversion exists
# (some pair changes relative order) AND Together does NOT cross (co-occurrence preserved,
# which separates a reorder from a branch retype). pm (ParallelMove) is a move -> in here;
# pl (ParallelToSequence) is a structural retype, NOT a relocation -> kept separate.
# cm (ConditionalMove) is a genuine composite -- conditional (Together crossing -> BRANCH) AND
# a move (Response inversion -> REORDER); it is a correct member of BOTH buckets.
REORDER_POSITIVE = {"sw", "sm", "pm", "cm"}


def _supp_delta(sig, rel) -> float:
    return _supp(sig.after, rel) - _supp(sig.before, rel)


def inversion_pairs(sig) -> set:
    """Unordered pairs {a,b} whose eventual order INVERTS: Response(a,b) weakens while
    Response(b,a) strengthens (via shifts). Both swaps and moves produce these."""
    # A dominance-flip variant was rejected: it drops pm (no dominant order to flip).
    changed = sig.born_relations | sig.died_relations | sig.shifted_relations
    resp = {(r[1], r[2]) for r in changed if r[0] == "Response" and r[1] != r[2]
            and r[1] in sig.present_both and r[2] in sig.present_both}
    out = set()
    for (a, b) in resp:
        if (b, a) in resp and _supp_delta(sig, ("Response", a, b)) < 0 \
                and _supp_delta(sig, ("Response", b, a)) > 0:
            out.add(frozenset((a, b)))
    return out


def reorder_sig(sig) -> bool:
    """REORDER = a mutual Response inversion exists AND Together does not cross (co-occurrence
    preserved -> a reorder, not a branch retype). Covers swap + serial-move + parallel-move."""
    if not inversion_pairs(sig):
        return False
    for rel in sig.born_relations | sig.died_relations:
        if rel[0] == "Together" and rel[1] in sig.present_both and rel[2] in sig.present_both:
            return False
    return True


def report_reorder(rows: list[dict]) -> None:
    by = _by_code(rows)

    def is_leftover(r):  # reaches this fork: claimed by none of OCC/BRANCH/SKIP
        return not r["miss"] and not r.get("occ") and not r.get("branch") and not r.get("skip")

    print("REORDER (swap + serial-move + parallel-move; order inversion, co-occurrence preserved)")
    print("  — LEFTOVER after OCC/BRANCH/SKIP\n")
    print(f"{'pat':5}{'REORDER':>9}{'left':>6}{'scored':>7}  inv-pairs(mean)  membership")
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        res = [r for r in rs if is_leftover(r)]
        ro = sum(1 for r in res if r.get("reorder"))
        ninv = [len(inversion_pairs(r["sig"])) for r in res if r.get("sig")]
        mean_inv = f"{sum(ninv)/len(ninv):.1f}" if ninv else "-"
        mem = "REORDER" if code in REORDER_POSITIVE else "neg"
        print(f"{code:5}{ro:>9}{len(res):>6}{scored:>7}  {mean_inv:>13}    {mem}")

    left = [r for r in rows if is_leftover(r)]
    tp = sum(1 for r in left if r["code"] in REORDER_POSITIVE and r.get("reorder"))
    npos = sum(1 for r in rows if not r["miss"] and r["code"] in REORDER_POSITIVE)
    fp = sum(1 for r in left if r["code"] not in REORDER_POSITIVE and r.get("reorder"))
    nneg = sum(1 for r in left if r["code"] not in REORDER_POSITIVE)
    print(f"\n  REORDER recall {tp}/{npos}   leftover-specificity {nneg-fp}/{nneg} (false-fires {fp})"
          f"   members={sorted(REORDER_POSITIVE)}")
    leftover_codes = Counter(r["code"] for r in left)
    print("  Leftover composition (what reaches this fork): "
          + ", ".join(f"{c}:{n}" for c, n in leftover_codes.most_common()))


# ---- SKIP bucket: an activity becomes optional (still present, but participates less) ----
# Signal: Existence(x,1) support DOWN (x in fewer cases, but STAYS positive -- not removal)
# AND a ChainResponse touching x (in or out) also DOWN. Frequency (cd/fr) lacks the Existence
# drop (it only redistributes Response routing); parallel/swap preserve x.
SKIP_POSITIVE = {"cb"}


def skip_sig(sig) -> bool:
    # shifted_relations only: a ChainResponse that dies outright is a removal signal already
    # claimed by OCC upstream. Measured on Ostovar: including died_relations costs 8 false
    # SKIP fires for 1 cb recall gain.
    changed = sig.shifted_relations
    down_exist = {rel[1] for rel in changed
                  if rel[0] == "Existence" and rel[2] == "1" and rel[1] in sig.present_both
                  and _supp(sig.after, rel) < _supp(sig.before, rel)}
    for x in down_exist:
        chain_down = any(rel[0] == "ChainResponse" and (rel[1] == x or rel[2] == x)
                         and _supp(sig.after, rel) < _supp(sig.before, rel)
                         for rel in changed)
        if chain_down:
            return True
    return False


def report_skip(rows: list[dict]) -> None:
    by = _by_code(rows)
    print("SKIP (Existence(x,1) DOWN-but-positive + ChainResponse(x,*) DOWN) — gate purity\n")
    print(f"{'pat':5}{'SK':>9}{'SK&resid':>10}{'scored':>8}  membership")
    tp = fn = fp = tn = 0
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        sk = sum(1 for r in rs if r.get("skip"))
        # residual = not already taken by OCC or BRANCH (SKIP runs third in the tree)
        sk_res = sum(1 for r in rs if r.get("skip") and not r.get("occ") and not r.get("branch"))
        pos = code in SKIP_POSITIVE
        if pos:
            tp += sk_res; fn += (scored - sk_res)
        else:
            fp += sk_res; tn += (scored - sk_res)
        print(f"{code:5}{f'{sk}/{scored}':>9}{f'{sk_res}/{scored}':>10}{scored:>8}  {'POS' if pos else 'neg'}")
    print(f"\nSkip detected (recall, residual) {tp}/{tp+fn}")
    print(f"Specificity (residual, post OCC+BRANCH): non-skip silent {tn}/{tn+fp}; false-fires {fp}")


# ---- FREQUENCY bucket: pure routing/probability change, no structural edit ----
# Positive signature: an activity's Response routing REDISTRIBUTES (some out-edges down, some
# up = conservation) with NO structural crossing (no born/died), no Existence drop, no order
# inversion. Separates a genuine frequency drift from both the structural buckets and (via the
# redistribution requirement) from a bare spurious shift.
FREQ_POSITIVE = {"cd", "fr"}


def freq_sig(sig) -> bool:
    # 1) no structural edit among already-present activities (pure shift, nothing born/died).
    # Existence relations put a cardinality level in rel[2], not an activity, so a born/died
    # Existence never trips this guard. Masked today because route() runs OCC first; revisit
    # if the gate order changes.
    for rel in sig.born_relations | sig.died_relations:
        if rel[1] in sig.present_both and rel[2] in sig.present_both:
            return False
    # 2) routing redistribution: some activity A has an out-Response down AND an out-Response up
    out_down: set = set()
    out_up: set = set()
    for rel in sig.shifted_relations:
        if rel[0] == "Response" and rel[1] != rel[2] and rel[1] in sig.present_both and rel[2] in sig.present_both:
            d = _supp_delta(sig, rel)
            if d < 0:
                out_down.add(rel[1])
            elif d > 0:
                out_up.add(rel[1])
    return bool(out_down & out_up)


def report_freq(rows: list[dict]) -> None:
    by = _by_code(rows)

    def is_leftover(r):  # reaches FREQUENCY: claimed by none of OCC/BRANCH/SKIP/REORDER
        return (not r["miss"] and not r.get("occ") and not r.get("branch")
                and not r.get("skip") and not r.get("reorder"))

    print("FREQUENCY (routing redistribution; no structural crossing) — LEFTOVER after OCC/BRANCH/SKIP/REORDER\n")
    print(f"{'pat':5}{'FREQ':>7}{'left':>6}{'scored':>7}  membership")
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        res = [r for r in rs if is_leftover(r)]
        fq = sum(1 for r in res if r.get("freq"))
        mem = "FREQ" if code in FREQ_POSITIVE else "neg"
        print(f"{code:5}{fq:>7}{len(res):>6}{scored:>7}  {mem}")

    left = [r for r in rows if is_leftover(r)]
    tp = sum(1 for r in left if r["code"] in FREQ_POSITIVE and r.get("freq"))
    npos = sum(1 for r in rows if not r["miss"] and r["code"] in FREQ_POSITIVE)
    fp = sum(1 for r in left if r["code"] not in FREQ_POSITIVE and r.get("freq"))
    nneg = sum(1 for r in left if r["code"] not in FREQ_POSITIVE)
    print(f"\n  FREQUENCY recall {tp}/{npos}   leftover-specificity {nneg-fp}/{nneg} (false-fires {fp})"
          f"   members={sorted(FREQ_POSITIVE)}")
    print("  Leftover composition: " + ", ".join(f"{c}:{n}" for c, n in Counter(r['code'] for r in left).most_common()))


def route(r) -> str:
    """Route a drift to its primary bucket via tree precedence (validated lenses only)."""
    if r.get("occ"):
        return "OCC"
    if r.get("branch"):
        return "BRANCH"
    if r.get("skip"):
        return "SKIP"
    if r.get("reorder"):
        return "REORDER"
    if r.get("freq"):
        return "FREQUENCY"
    return "OTHER"   # residual: pl + recall-gap stragglers


# Final-leaf membership (which benchmark labels belong to each leaf). re is bidirectional
# (insert on add, remove on revert) -> member of both. cm is conditional+move -> BRANCH & REORDER.
# Composites belong to COMPOSITE (a composite routed to a single sub-op is a real miss).
LEAF_MEMBERS = {
    "DUPLICATION": {"cp"}, "LOOP": {"lp"}, "REMOVAL": {"cre", "pre", "re"}, "INSERTION": {"re"},
    "SUBSTITUTION": {"rp"}, "COMPOSITE": {"ior", "iro", "oir", "ori", "rio", "roi"},
    "BRANCH": {"cf", "cm"}, "SKIP": {"cb"}, "REORDER": {"sw", "sm", "pm", "cm"}, "FREQUENCY": {"cd", "fr"},
}


def final_leaf(r, floor: float) -> str:
    b = route(r)
    return occ_sublabel(r["sig"], floor) if b == "OCC" else b


def report_leaves(rows: list[dict], floor: float) -> None:
    scored = [r for r in rows if not r["miss"]]
    leaves = ["LOOP", "DUPLICATION", "REMOVAL", "INSERTION", "SUBSTITUTION", "COMPOSITE",
              "BRANCH", "SKIP", "REORDER", "FREQUENCY"]
    print(f"Final-leaf precision/recall/F1 over {len(scored)} scored drifts ({sum(1 for r in rows if r['miss'])} det-miss)\n")
    print(f"  {'leaf':13}{'routed':>7}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    labelled = [(r, final_leaf(r, floor)) for r in scored]
    for L in leaves:
        mem = LEAF_MEMBERS[L]
        routed = [r for r, leaf in labelled if leaf == L]
        tp = sum(1 for r in routed if r["code"] in mem)
        fp = len(routed) - tp
        npos = sum(1 for r in scored if r["code"] in mem)
        fn = npos - tp
        p = tp / len(routed) if routed else 0.0
        rec = tp / npos if npos else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        print(f"  {L:13}{len(routed):>7}{tp:>4}{fp:>4}{fn:>4}{p:>7.2f}{rec:>7.2f}{f1:>7.2f}   [{','.join(sorted(mem))}]")


def report_summary(rows: list[dict]) -> None:
    members = {"OCC": OCC_POSITIVE, "BRANCH": BRANCH_POSITIVE, "SKIP": SKIP_POSITIVE,
               "REORDER": REORDER_POSITIVE, "FREQUENCY": FREQ_POSITIVE}
    by = _by_code(rows)

    print("Per-pattern routing (tree precedence OCC -> BRANCH -> OTHER):")
    print(f"  {'pat':5}{'scored':>7}  routed-bucket distribution")
    for code in sorted(by):
        rs = [r for r in by[code] if not r["miss"]]
        if not rs:
            print(f"  {code:5}{0:>7}  (all det-miss)"); continue
        dist = Counter(route(r) for r in rs)
        print(f"  {code:5}{len(rs):>7}  " + ", ".join(f"{b}:{c}" for b, c in dist.most_common()))

    print("\nValidated buckets (recall over member patterns, specificity over non-members):")
    for bucket, mem in members.items():
        tp = sum(1 for r in rows if not r["miss"] and r["code"] in mem and route(r) == bucket)
        pos = sum(1 for r in rows if not r["miss"] and r["code"] in mem)
        # specificity: non-members that did NOT route here
        fp = sum(1 for r in rows if not r["miss"] and r["code"] not in mem and route(r) == bucket)
        neg = sum(1 for r in rows if not r["miss"] and r["code"] not in mem)
        print(f"  {bucket:7} recall {tp}/{pos}   specificity {neg-fp}/{neg} (false-fires {fp})   members={sorted(mem)}")

    other = sorted({r["code"] for r in rows if not r["miss"] and route(r) == "OTHER"})
    print(f"\nOTHER (residual, no validated lens yet): patterns landing here = {other}")

    # Per-bucket TP/FP confusion (full tree, single routing per drift).
    print("\nPer-bucket confusion (full tree routing; each drift -> exactly one bucket):")
    print(f"  {'bucket':9}{'routed':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}")
    scored = [r for r in rows if not r["miss"]]
    for bucket, mem in members.items():
        routed = [r for r in scored if route(r) == bucket]
        tp = sum(1 for r in routed if r["code"] in mem)
        fp = len(routed) - tp
        npos = sum(1 for r in scored if r["code"] in mem)
        fn = npos - tp
        prec = f"{tp/len(routed):.2f}" if routed else "-"
        rec = f"{tp/npos:.2f}" if npos else "-"
        print(f"  {bucket:9}{len(routed):>7}{tp:>5}{fp:>5}{fn:>5}{prec:>11}{rec:>9}")
    n_other = sum(1 for r in scored if route(r) == "OTHER")
    print(f"  {'OTHER':9}{n_other:>7}{'-':>5}{'-':>5}{'-':>5}{'-':>11}{'-':>9}  (unbucketed residual)")
    print(f"  total scored drifts: {len(scored)}; det-misses: {sum(1 for r in rows if r['miss'])}")


def _auto_cfg(traces, args):
    return resolve_config(len(traces), args.batch_size, args.span)


def _n_order(series) -> int:
    """Number of tracked ORDER relations (the ones the detector fires on)."""
    return sum(1 for r in (set(series.support) | set(series.fulfillments))
               if r[0] in ("ChainResponse", "Response"))


def _relative_filter(cps, series, frac: float):
    """RELATIVE intensity filter: keep a change point only if it shifted at least
    `frac` of the tracked order-relations. Normalising by the relation count makes the cut
    transferable across processes of different density (Ostovar ~500 relations vs Ceravolo ~50).
    frac<=0 disables it (recall-first default)."""
    if frac <= 0:
        return cps
    thr = frac * max(1, _n_order(series))
    return [c for c in cps if c.intensity >= thr]


def _series_events(traces, channels, args):
    B, span = _auto_cfg(traces, args)
    _, manifest, support, fulf, act = build_series(traces, B, {"chainresponse", "response", "existence", "choice"})
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"])) for x in manifest]
    series = Series(batches, support, fulf, act)
    cps = sequential_detect(series, channels, args.test, args.confidence, span, templates={"ChainResponse", "Response"})
    cps = _relative_filter(cps, series, args.min_int_frac)
    return series, events_of(cps), batches


def _detect_events(traces, channels, args):
    """Detected change points clustered into events, KEEPING each event's peak intensity
    (max # relations that shifted across the cluster). Returns ([(onset_case, intensity)], B, N)
    UNFILTERED -- the relative filter is applied/swept in the reports so we can see its effect."""
    B, span = _auto_cfg(traces, args)
    _, manifest, support, fulf, act = build_series(traces, B, {"chainresponse", "response", "existence", "choice"})
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"])) for x in manifest]
    series = Series(batches, support, fulf, act)
    cps = sequential_detect(series, channels, args.test, args.confidence, span, templates={"ChainResponse", "Response"})
    events: list[tuple[int, int]] = []
    prev_b = None
    for c in sorted(cps, key=lambda cp: cp.boundary):
        if prev_b is not None and c.boundary - prev_b <= 1:
            events[-1] = (events[-1][0], max(events[-1][1], c.intensity))
        else:
            events.append((c.case_index, c.intensity))
        prev_b = c.boundary
    return events, B, _n_order(series)


def _sig_for(series, events, batches, true_cp, tol, channels, args):
    """Signature of the detected event nearest true_cp, or None on detection miss."""
    if not events:
        return None
    near = min(range(len(events)), key=lambda i: abs(batches[events[i][0]].start_case - true_cp))
    if abs(batches[events[near][0]].start_case - true_cp) > tol:
        return None
    fb, lb = events[near]
    b_lo = events[near - 1][1] if near > 0 else 0
    a_hi = events[near + 1][0] if near + 1 < len(events) else len(batches)
    return build_change_signature(build_evidence(series, b_lo, fb, lb, a_hi, channels, args.test, args.confidence))


def iter_ostovar(args, channels):
    """Yield (code, signature-or-None) per true drift (two per log)."""
    for path in sorted(args.logs_dir.glob("*.xes.gz")):
        m = NAME_RE.match(path.name)
        if not m or m.group(1) not in PATTERN_MAP:
            continue
        code = PATTERN_MAP[m.group(1)]
        series, events, batches = _series_events(parse_traces(path, "concept:name"), channels, args)
        for true_cp in TRUE_CPS:
            yield code, _sig_for(series, events, batches, true_cp, args.tolerance, channels, args)


def iter_ceravolo(args, channels):
    """Yield (label, signature-or-None) per true drift (one per log)."""
    gt = load_ground_truth(args.results_csv)
    keep = {s.strip() for s in args.sizes.split(",")} if args.sizes else set(SIZES)
    for path in sorted(args.ceravolo_dir.glob("*.xes.gz")):
        base = log_base_name(path)
        m = CER_NAME_RE.search(base)
        if not m:
            continue
        size, label = m.group(1), m.group(2).lower()
        if size not in SIZES or size not in keep:
            continue
        c = gt.get(base, int(size) // 2 - 1)
        series, events, batches = _series_events(parse_traces(path, "concept:name"), channels, args)
        # absolute window (--cer-tol) overrides the fractional one; at 1000 traces the 15%
        # frac is only 150 cases -- widen to ~200 (Ostovar parity) to recover det-misses.
        tol = args.cer_tol if args.cer_tol > 0 else max(1, round(args.tolerance_frac * int(size)))
        yield label, _sig_for(series, events, batches, c, tol, channels, args)


def iter_detect(args, channels):
    """Detection-only pass: per true drift, distance (cases) to the nearest detected event
    onset (None if the log produced no events), the log's event count, and the match window."""
    def logs(kind):
        if kind == "ostovar":
            for path in sorted(args.logs_dir.glob("*.xes.gz")):
                m = NAME_RE.match(path.name)
                if m and m.group(1) in PATTERN_MAP:
                    yield path, PATTERN_MAP[m.group(1)], TRUE_CPS, args.tolerance
        else:
            gt = load_ground_truth(args.results_csv)
            keep = {s.strip() for s in args.sizes.split(",")} if args.sizes else set(SIZES)
            for path in sorted(args.ceravolo_dir.glob("*.xes.gz")):
                base = log_base_name(path); m = CER_NAME_RE.search(base)
                if not m or m.group(1) not in SIZES or m.group(1) not in keep:
                    continue
                size = m.group(1)
                tol = args.cer_tol if args.cer_tol > 0 else max(1, round(args.tolerance_frac * int(size)))
                yield path, m.group(2).lower(), [gt.get(base, int(size) // 2 - 1)], tol
    kinds = ["ostovar", "ceravolo"] if args.dataset == "both" else [args.dataset]
    for kind in kinds:
        for path, code, true_cps, tol in logs(kind):
            events, B, N = _detect_events(parse_traces(path, "concept:name"), channels, args)
            yield {"ds": kind, "code": code, "events": events, "actual": sorted(true_cps),
                   "lag": tol, "B": B, "N": N, "frac": args.min_int_frac}


def _tp_fp_lags(detected, actual, lag):
    """Greedy nearest-unused matcher. Equivalent to cdrift's LP getTP_FP when the per-actual
    lag windows are disjoint (Ostovar tol<=500, single-CP Ceravolo); it can under-count TP when
    they overlap. Returns (TP, FP, FN, matched-pair lags)."""
    assigned: set[int] = set()
    tp, lags = 0, []
    for cp in actual:
        cands = [(abs(d - cp), j) for j, d in enumerate(detected)
                 if abs(d - cp) <= lag and j not in assigned]
        if cands:
            dist, best = min(cands)          # `detected` is sorted, so index order == value order
            assigned.add(best); tp += 1; lags.append(dist)
    return tp, len(detected) - len(assigned), len(actual) - tp, lags


def _uniform(values) -> str:
    """Header display for a per-record parameter: the single value if every record agrees,
    else the observed range. Keeps mixed-tolerance runs honest instead of showing record 0."""
    vs = set(values)
    return str(next(iter(vs))) if len(vs) == 1 else f"{min(vs)}-{max(vs)}"


def report_detect(records: list[dict]) -> None:
    print("\nMethod: cdrift benchmark (Adams et al., 'An Experimental Evaluation of Process Concept")
    print("Drift Detection') -- F1-Score + Average Lag. A detected CP is a TP if within `lag` of an")
    print("actual CP (nearest unused match); Average Lag = mean |detected-actual| over matched pairs.")
    for ds in ("ostovar", "ceravolo"):
        rs = [r for r in records if r["ds"] == ds]
        if not rs:
            continue
        TP = FP = FN = 0
        lags: list[int] = []
        for r in rs:
            thr = r["frac"] * max(1, r["N"])          # relative-intensity filter (--min-int-frac)
            detected = sorted(c for c, i in r["events"] if i >= thr)
            tp, fp, fn, lg = _tp_fp_lags(detected, r["actual"], r["lag"])
            TP += tp; FP += fp; FN += fn; lags += lg
        b_mean = sum(r["B"] for r in rs) / len(rs)
        lag_disp = _uniform(rec_["lag"] for rec_ in rs)
        batch_disp = _uniform(rec_["B"] for rec_ in rs)
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"\n=== {ds.upper()} — detection (lag={lag_disp} cases, batch≈{batch_disp}) ===")
        print(f"  actual CPs: {TP+FN}   matched (TP): {TP}   FP: {FP}   FN: {FN}")
        print(f"  Precision {prec:.2f}   Recall {rec:.2f}   F1-Score {f1:.2f}")
        if lags:
            print(f"  AVERAGE LAG (matched pairs): {statistics.mean(lags):.1f} cases  "
                  f"(median {statistics.median(lags):.0f}, max {max(lags)})  -- i.e. ~{statistics.mean(lags)/b_mean:.1f} batch")
        print(f"  note: detector over-fires (~{statistics.mean(len(r['events']) for r in rs):.1f} detected CPs/log), "
              f"which lowers precision/F1 while recall & lag stay strong.")


def classify_sig(sig, floor: float) -> str:
    """Route a single change signature to its final leaf (OCC sub-leaf / BRANCH / SKIP /
    REORDER / FREQUENCY / OTHER), reusing the decision-tree predicates."""
    r = {"occ": bool(occ_shape(sig, floor)), "branch": branch_tc(sig), "skip": skip_sig(sig),
         "reorder": reorder_sig(sig), "freq": freq_sig(sig), "sig": sig}
    b = route(r)
    return occ_sublabel(sig, floor) if b == "OCC" else b


# Leaves that count as a CONFIDENT structural label: such a CP is confirmed regardless of
# intensity. FREQUENCY is the validity-residual and OTHER is unlabelable -> NOT confident.
_STRUCTURAL = {"LOOP", "DUPLICATION", "REMOVAL", "INSERTION", "SUBSTITUTION", "COMPOSITE",
               "BRANCH", "SKIP", "REORDER"}


def _detect_events_full(traces, channels, args):
    """Per log: (series, batches, events, N) where each event = (fb, lb, peak_intensity) in
    batch indices -- enough to rebuild the labeling windows. UNFILTERED."""
    B, span = _auto_cfg(traces, args)
    _, manifest, support, fulf, act = build_series(traces, B, {"chainresponse", "response", "existence", "choice"})
    batches = [Batch(int(x["batch_index"]), int(x["start_case"]), int(x["end_case"]), int(x["n_cases"])) for x in manifest]
    series = Series(batches, support, fulf, act)
    cps = sequential_detect(series, channels, args.test, args.confidence, span, templates={"ChainResponse", "Response"})
    events: list[list[int]] = []
    for c in sorted(cps, key=lambda cp: cp.boundary):
        if events and c.boundary - events[-1][1] <= 1:
            events[-1][1] = c.boundary
            events[-1][2] = max(events[-1][2], c.intensity)
        else:
            events.append([c.boundary, c.boundary, c.intensity])
    return series, batches, events, _n_order(series)


def iter_confirm(args, channels):
    """Per log: label EVERY detected event using full-regime windows and record its leaf,
    onset, intensity. Yields {ds, code, events:[(onset, inten, leaf)], actual, lag, N}."""
    def logs(kind):
        if kind == "ostovar":
            for path in sorted(args.logs_dir.glob("*.xes.gz")):
                m = NAME_RE.match(path.name)
                if m and m.group(1) in PATTERN_MAP:
                    yield path, PATTERN_MAP[m.group(1)], TRUE_CPS, args.tolerance
        else:
            gt = load_ground_truth(args.results_csv)
            keep = {s.strip() for s in args.sizes.split(",")} if args.sizes else set(SIZES)
            for path in sorted(args.ceravolo_dir.glob("*.xes.gz")):
                base = log_base_name(path); m = CER_NAME_RE.search(base)
                if not m or m.group(1) not in SIZES or m.group(1) not in keep:
                    continue
                size = m.group(1)
                tol = args.cer_tol if args.cer_tol > 0 else max(1, round(args.tolerance_frac * int(size)))
                yield path, m.group(2).lower(), [gt.get(base, int(size) // 2 - 1)], tol
    floor = 1.0 - args.born_k
    kinds = ["ostovar", "ceravolo"] if args.dataset == "both" else [args.dataset]
    unlabelable: Counter = Counter()
    for kind in kinds:
        for path, code, true_cps, tol in logs(kind):
            series, batches, events, N = _detect_events_full(parse_traces(path, "concept:name"), channels, args)
            out = []
            for i, (fb, lb, inten) in enumerate(events):
                # _detect_events_full only opens a new event when the gap exceeds one batch,
                # so fb > b_lo and a_hi > lb hold by construction: every window is non-degenerate.
                b_lo = events[i - 1][1] if i > 0 else 0
                a_hi = events[i + 1][0] if i + 1 < len(events) else len(batches)
                try:
                    sig = build_change_signature(build_evidence(series, b_lo, fb, lb, a_hi, channels, args.test, args.confidence))
                    leaf = classify_sig(sig, floor)
                except (ArithmeticError, LookupError) as exc:
                    # "OTHER" is not in _STRUCTURAL, so a swallowed error silently changes the
                    # dual-gate verdict. Fall back, but never silently.
                    unlabelable[type(exc).__name__] += 1
                    leaf = "OTHER"
                out.append((batches[fb].start_case, inten, leaf))
            yield {"ds": kind, "code": code, "events": out, "actual": sorted(true_cps), "lag": tol, "N": N}
    for exc_name, n in sorted(unlabelable.items()):
        print(f"[warn] {n} events unlabelable ({exc_name})", file=sys.stderr)


def report_confirm(records: list[dict], frac: float) -> None:
    """Compare confirmation gates on detection precision AND end-to-end (detected + correctly
    labeled) -- showing the dual gate (intensity>=frac OR structural label) recovers labeling."""
    inv = {}                       # code -> set of acceptable leaves
    for leaf, codes in LEAF_MEMBERS.items():
        for c in codes:
            inv.setdefault(c, set()).add(leaf)

    for ds in ("ostovar", "ceravolo"):
        rs = [r for r in records if r["ds"] == ds]
        if not rs:
            continue
        print(f"\n=== {ds.upper()} — confirmation gates (frac={frac}, "
              f"lag={_uniform(r['lag'] for r in rs)}) ===")
        print(f"  {'gate':22}{'det-P':>7}{'det-R':>7}{'det-F1':>8}{'e2e-correct':>13}")
        for kind in ("all", "intensity>=1%", "dual: int OR label"):
            TP = FP = FN = 0
            e2e_ok = 0
            n_actual = 0
            for r in rs:
                N = max(1, r["N"])
                def confirmed(ev):
                    onset, inten, leaf = ev
                    if kind == "all":
                        return True
                    strong = inten >= frac * N
                    if kind == "intensity>=1%":
                        return strong
                    return strong or (leaf in _STRUCTURAL)        # dual
                conf = [ev for ev in r["events"] if confirmed(ev)]
                det_cases = sorted(c for c, _i, _l in conf)
                tp, fp, fn, _ = _tp_fp_lags(det_cases, r["actual"], r["lag"])
                TP += tp; FP += fp; FN += fn
                # end-to-end: each true CP -> nearest confirmed event within lag -> correct leaf?
                # NB: unlike _tp_fp_lags, this nearest-search has no assignment bookkeeping --
                # at --tolerance >= 500 the Ostovar windows overlap and one event can score both CPs.
                for cp in r["actual"]:
                    n_actual += 1
                    near = min(conf, key=lambda ev: abs(ev[0] - cp), default=None)
                    if near is not None and abs(near[0] - cp) <= r["lag"] and near[2] in inv.get(r["code"], set()):
                        e2e_ok += 1
            p = TP / (TP + FP) if TP + FP else 0.0
            rec = TP / (TP + FN) if TP + FN else 0.0
            f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
            print(f"  {kind:22}{p:>7.2f}{rec:>7.2f}{f1:>8.2f}{f'{e2e_ok}/{n_actual} ({e2e_ok/n_actual:.2f})':>13}")
        unreachable = sorted({r["code"] for r in rs} - set(inv))
        print(f"  (codes with no expected leaf, always counted wrong: {unreachable})")


def report_intensity(records: list[dict]) -> None:
    """RELATIVE intensity filter. Score each detection by intensity/N (fraction of
    tracked order-relations it shifted) so the cut transfers across process densities. Reports
    TRUE-vs-SPURIOUS fraction distributions, separability AUC, and a sweep of the min-fraction
    cut -> precision/recall/F1, flagging (i) best F1 and (ii) the largest cut that KEEPS full
    recall (= misses nothing)."""
    for ds in ("ostovar", "ceravolo"):
        rs = [r for r in records if r["ds"] == ds]
        if not rs:
            continue
        tp_fr, fp_fr = [], []                          # intensity FRACTION of matched vs spurious
        for r in rs:
            N = max(1, r["N"])
            for case, inten in r["events"]:
                near = min((abs(case - a) for a in r["actual"]), default=10 ** 9)
                (tp_fr if near <= r["lag"] else fp_fr).append(inten / N)
        if not (tp_fr and fp_fr):
            continue
        print(f"\n=== {ds.upper()} — RELATIVE intensity (fraction of tracked relations, "
              f"lag={_uniform(r['lag'] for r in rs)}) ===")
        print(f"  TRUE (n={len(tp_fr)}):     fraction  mean {statistics.mean(tp_fr):.3f}  median {statistics.median(tp_fr):.3f}  min {min(tp_fr):.3f}")
        print(f"  SPURIOUS (n={len(fp_fr)}): fraction  mean {statistics.mean(fp_fr):.3f}  median {statistics.median(fp_fr):.3f}  max {max(fp_fr):.3f}")
        wins = sum(1 for t in tp_fr for f in fp_fr if t > f) + 0.5 * sum(1 for t in tp_fr for f in fp_fr if t == f)
        print(f"  separability AUC (P[true > spurious]): {wins/(len(tp_fr)*len(fp_fr)):.2f}")

        def metrics(frac):
            TP = FP = FN = 0
            for r in rs:
                thr = frac * max(1, r["N"])
                det = sorted(c for c, i in r["events"] if i >= thr)
                tp, fp, fn, _ = _tp_fp_lags(det, r["actual"], r["lag"])
                TP += tp; FP += fp; FN += fn
            p = TP / (TP + FP) if TP + FP else 0.0
            rec = TP / (TP + FN) if TP + FN else 0.0
            return p, rec, (2 * p * rec / (p + rec) if p + rec else 0.0), TP, FP, FN

        fracs = [round(x, 3) for x in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]]
        swept = {f: metrics(f) for f in fracs}
        base_rec = swept[0.0][1]
        print(f"  {'min-fraction':>13}{'P':>7}{'R':>7}{'F1':>7}{'FN':>5}")
        best_f1 = max(fracs, key=lambda f: swept[f][2])
        keep_rec = max((f for f in fracs if swept[f][1] >= base_rec - 1e-9), default=0.0)  # largest no-miss cut
        for f in fracs:
            p, rec, f1, TP, FP, FN = swept[f]
            mark = ""
            if f == best_f1: mark += "  <- best F1"
            if f == keep_rec: mark += "  <- max cut w/ NO misses"
            print(f"  {f:>13.3f}{p:>7.2f}{rec:>7.2f}{f1:>7.2f}{FN:>5}{mark}")


# ---- BRANCH bucket: Together-crossing (the dataset-agnostic branch signal) ----
# A branch restructuring = two ALREADY-present activities start/stop co-occurring:
# Together(a,b) crosses 0<->positive. (ExclusiveChoice-crossing was the Ostovar-only
# shadow of this.) Conditional family. cd is frequency (Response shift, no crossing).
BRANCH_POSITIVE = {"cf", "cm"}


def branch_tc(sig) -> bool:
    """Together(a,b) crosses 0<->positive for a pair of already-present activities.

    Decision: `present_both` is load-bearing, not tidiness. Together is derived as
    Existence(a,1) + Existence(b,1) - Choice(a,b), so for a mandatory `a` it collapses to
    Together(a,x) = P(x) -- a pure presence test, the same defect that made
    ExclusiveChoice(a,x) = 1 - P(x) fire on every skip (see l4_sublens/ECg). It stays out of
    reach only because present_both is an exact set-membership test; loosening it to a
    born-floor test, as occ_shape does for diluted insertions, silently reopens that bug.
    """
    for rel in sig.born_relations | sig.died_relations:
        if rel[0] == "Together" and rel[1] != rel[2] \
                and rel[1] in sig.present_both and rel[2] in sig.present_both:
            return True
    return False


def report_branch(rows: list[dict]) -> None:
    by = _by_code(rows)
    print("BRANCH (Together-crossing: two present activities start/stop co-occurring) — gate purity\n")
    print(f"{'pat':5}{'TC':>9}{'TC&!OCC':>10}{'scored':>8}  membership")
    tp = fn = fp = tn = 0
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        tc = sum(1 for r in rs if r.get("branch"))
        tc_post = sum(1 for r in rs if r.get("branch") and not r.get("occ"))
        pos = code in BRANCH_POSITIVE
        if pos:
            tp += tc_post; fn += (scored - tc_post)
        else:
            fp += tc_post; tn += (scored - tc_post)
        print(f"{code:5}{f'{tc}/{scored}':>9}{f'{tc_post}/{scored}':>10}{scored:>8}  {'POS' if pos else 'neg'}")
    print(f"\nBranch detected (recall, post-OCC) {tp}/{tp+fn}")
    print(f"Specificity POST-OCC: non-branch silent {tn}/{tn+fp}; false-fires {fp}")


COND_POSITIVE = {"cf", "cm", "cd"}


def report_cond(rows: list[dict]) -> None:
    by = _by_code(rows)

    print("CONDITIONAL (gated ExclusiveChoice: both endpoints toggle) — gate purity\n")
    print(f"{'pat':5}{'ECg':>9}{'ECg&!OCC':>11}{'scored':>8}  membership")
    # raw = over all drifts; post = only drifts that did NOT fire OCC (what reaches this node)
    tp = fn = fp_raw = tn_raw = fp_post = tn_post = 0
    for code in sorted(by):
        rs = by[code]
        scored = sum(1 for r in rs if not r["miss"])
        ecg = sum(1 for r in rs if r.get("cond"))
        ecg_post = sum(1 for r in rs if r.get("cond") and not r.get("occ"))
        pos = code in COND_POSITIVE
        if pos:
            tp += ecg_post; fn += (scored - ecg_post)
        else:
            fp_raw += ecg; tn_raw += (scored - ecg)
            fp_post += ecg_post; tn_post += (scored - ecg_post)
        print(f"{code:5}{f'{ecg}/{scored}':>9}{f'{ecg_post}/{scored}':>11}{scored:>8}  {'POS' if pos else 'neg'}")
    print(f"\nConditional detected (recall, post-OCC) {tp}/{tp+fn}")
    print(f"Specificity RAW (all drifts):     non-conditional silent {tn_raw}/{tn_raw+fp_raw}; false-fires {fp_raw}")
    print(f"Specificity POST-OCC (tree order): non-conditional silent {tn_post}/{tn_post+fp_post}; false-fires {fp_post}")
    print("  (post-OCC = the residual that actually reaches this node, since OCC runs first)")


def _rel_str(rel, sig) -> str:
    t, a, b = rel
    sb = sig.before.relation_support.get(rel, 0.0) if sig.before else 0.0
    sa = sig.after.relation_support.get(rel, 0.0) if sig.after else 0.0
    return f"{t}({a},{b}) {sb:.2f}->{sa:.2f}"


def report_occmiss(rows: list[dict], codes: set[str]) -> None:
    """Diagnose WHY OCC missed these drifts: dump each activity's Existence(x,1) before->after
    so we can see dilution (~0.02->0.9), already-present (duplication), or no presence change."""
    sel = [r for r in rows if r["code"] in codes and not r["miss"] and not r.get("occ")]
    print(f"OCC-MISSED drifts for {sorted(codes)}: {len(sel)} (routed elsewhere)\n")
    tally = Counter()
    for i, r in enumerate(sel):
        sig = r["sig"]
        present_before = sig.before.activities
        ex = []
        for rel in sorted(set(sig.before.relation_support) | set(sig.after.relation_support)):
            if rel[0] == "Existence" and rel[2] == "1":
                b = sig.before.relation_support.get(rel, 0.0)
                a = sig.after.relation_support.get(rel, 0.0)
                ex.append((rel[1], b, a))
        ex.sort(key=lambda t: (-(t[2] - t[1]), t[0]))        # biggest INCREASE first, ties by name
        top = ex[:5]
        print(f"--- {r['code']} #{i}  born_act={sorted(sig.born_activities)} died_act={sorted(sig.died_activities)}")
        for x, b, a in top:
            if a - b <= 0:
                continue
            if b == 0.0:
                tag = "BORN(0->+) -- should fire OCC!"
                tally["born-not-fired"] += 1
            elif b < 0.20 * a:
                tag = f"DILUTED (x in before-regime? {x in present_before})"
                tally["diluted"] += 1
            else:
                tag = "already-present (dup-like / no clean insertion)"
                tally["present"] += 1
            print(f"     Existence({x},1) {b:.3f} -> {a:.3f}   {tag}")
        print()
    print(f"Tally over the top-5 increasing activities of each OCC-missed drift: {dict(tally)}")


def substitution_mirror(sig, floor: float) -> bool:
    """SUBSTITUTION (vs insert+remove composite): the born activity x takes over the dead
    activity y's exact slot -- a DIRECT EDGE TRANSFERS. For the same predecessor p,
    ChainResponse(p,y) is in DIED and ChainResponse(p,x) is in BORN; or for the same successor
    s, ChainResponse(y,s) DIED and ChainResponse(x,s) BORN. Threshold-free (gated only by the
    significance test behind born/died -- no magnitude/frequency cutoff). The transferred-edge
    test is what stops a global hub (DRIFT_PO) faking a substitution."""
    new_acts, gone_acts = occ_born_gone(sig, floor)
    if not (new_acts and gone_acts):
        return False
    born, died = sig.born_relations, sig.died_relations
    for x in new_acts:
        pred_x = {r[1] for r in born if r[0] == "ChainResponse" and r[2] == x}
        succ_x = {r[2] for r in born if r[0] == "ChainResponse" and r[1] == x}
        if not pred_x and not succ_x:
            continue
        for y in gone_acts:
            pred_y = {r[1] for r in died if r[0] == "ChainResponse" and r[2] == y}
            succ_y = {r[2] for r in died if r[0] == "ChainResponse" and r[1] == y}
            if (pred_x & pred_y) or (succ_x & succ_y):   # a direct neighbour edge transferred y -> x
                return True
    return False


def occ_sublabel(sig, born_floor: float) -> str:
    """Ordered OCC sub-classifier (peel most-distinctive first):
    1 CARDINALITY (single DUP/LOOP)  2 REMOVAL (single GONE)  3 INSERTION (single NEW)
    4 SUBSTITUTION (NEW+GONE only)   else COMPOSITE (>=2 independent occurrence ops)."""
    sh = occ_shape(sig, born_floor)
    core = set()
    if "NEW" in sh:
        core.add("INS")
    if "GONE" in sh:
        core.add("REM")
    if "LOOP" in sh:
        core.add("LOOP")
    if "DUP" in sh:
        core.add("DUP")
    if not core:
        raise ValueError("occ_sublabel called on a non-OCC signature")
    if len(core) == 1:
        return {"LOOP": "LOOP", "DUP": "DUPLICATION", "REM": "REMOVAL", "INS": "INSERTION"}[next(iter(core))]
    if core == {"INS", "REM"}:
        # NEW+GONE: substitution if the born activity mirrors the dead one's context, else
        # an independent insert+remove composite.
        return "SUBSTITUTION" if substitution_mirror(sig, born_floor) else "COMPOSITE"
    return "COMPOSITE"             # >=2 ops incl. cardinality


# Operation-level ground truth for OCC drifts (permissive: bidirectional / composite labels
# legitimately satisfy several occurrence operations).
_COMPO = {"INSERTION", "REMOVAL", "DUPLICATION", "LOOP", "COMPOSITE"}   # composite != substitution
EXP_OCC_OPS = {
    "cp": {"DUPLICATION"}, "lp": {"LOOP"},
    "re": {"INSERTION", "REMOVAL", "DUPLICATION", "LOOP"},   # Ostovar add/remove; Ceravolo re = dup-like
    "cre": {"REMOVAL", "INSERTION"}, "pre": {"REMOVAL", "INSERTION"},
    "rp": {"SUBSTITUTION", "INSERTION", "REMOVAL"},
    "ior": _COMPO, "iro": _COMPO, "oir": _COMPO, "ori": _COMPO, "rio": _COMPO, "roi": _COMPO,
}
_LAYER_OP = {"LOOP": "LOOP", "DUPLICATION": "DUPLICATION", "REMOVAL": "REMOVAL",
             "INSERTION": "INSERTION", "SUBSTITUTION": "SUBSTITUTION", "COMPOSITE": "COMPOSITE"}


def report_occsplit(rows: list[dict], born_floor: float) -> None:
    """Within OCC: ordered sub-layers, scored at the OPERATION level -- benchmark labels are
    bidirectional/composite, so scoring by label alone understates the separation."""
    occ_rows = [r for r in rows if not r["miss"] and r.get("occ")]
    layers = ["DUPLICATION", "LOOP", "REMOVAL", "INSERTION", "SUBSTITUTION", "COMPOSITE"]
    by_layer: dict[str, Counter] = {L: Counter() for L in layers}
    for r in occ_rows:
        by_layer[occ_sublabel(r["sig"], born_floor)][r["code"]] += 1
    print(f"OCC sub-layers (ordered peel) over {len(occ_rows)} OCC-bucketed drifts:\n")
    print(f"  {'layer':13}{'n':>4}{'op-correct':>12}{'op-prec':>9}   labels")
    for L in layers:
        c = by_layer[L]
        if not c:
            continue
        total = sum(c.values())
        op = _LAYER_OP.get(L)
        ok = sum(n for code, n in c.items() if op in EXP_OCC_OPS.get(code, set()))
        prec = ok / total if total else 0.0
        print(f"  {L:13}{total:>4}{ok:>12}{prec:>9.2f}   " + ", ".join(f"{k}:{v}" for k, v in c.most_common()))


def report_explore(rows: list[dict], codes: set[str]) -> None:
    """Dump the actual changed relations (by template & kind) for the given patterns,
    so we can SEE what the drift changes in Declare terms."""
    sel = [r for r in rows if r["code"] in codes and not r["miss"]]
    print(f"EXPLORE patterns={sorted(codes)} | {len(sel)} scored drifts\n")

    agg: Counter = Counter()   # (template, kind) -> count across drifts
    for i, r in enumerate(sel):
        sig = r["sig"]
        print(f"--- drift #{i} [{r['code']}]  occ={r['occfp']}  ECg={r['cond']} ---")
        for kind, rels in (("BORN", sig.born_relations), ("DIED", sig.died_relations), ("SHIFT", sig.shifted_relations)):
            if not rels:
                continue
            for rel in sorted(rels):
                agg[(rel[0], kind)] += 1
            shown = sorted(rels)[:12]
            print(f"  {kind:5}: " + "; ".join(_rel_str(rel, sig) for rel in shown)
                  + (f"  (+{len(rels)-12} more)" if len(rels) > 12 else ""))
        print()

    print("Aggregate (template x kind) across these drifts:")
    templates = sorted({t for t, _ in agg})
    print(f"  {'template':16}{'BORN':>6}{'DIED':>6}{'SHIFT':>7}")
    for t in templates:
        print(f"  {t:16}{agg[(t,'BORN')]:>6}{agg[(t,'DIED')]:>6}{agg[(t,'SHIFT')]:>7}")


def _no_logs() -> int:
    """Bad --logs-dir / --sizes / --dataset used to yield a well-formed all-zeros report."""
    print("no drifts scored -- check --logs-dir/--ceravolo-dir/--sizes/--dataset", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ostovar", "ceravolo", "both"], default="both")
    ap.add_argument("--logs-dir", type=Path, default=Path("cdrift-evaluation/EvaluationLogs/Ostovar"))
    ap.add_argument("--ceravolo-dir", type=Path, default=Path("cdrift-evaluation/EvaluationLogs/Ceravolo"))
    ap.add_argument("--results-csv", type=Path, default=Path("cdrift-evaluation/algorithm_results.csv"))
    ap.add_argument("--channels", default="support,confidence")
    ap.add_argument("--test", choices=["z", "chebyshev", "fisher"], default="z")
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--born-k", type=float, default=0.99, help="K_born; BORN if before-presence < (1-K_born). Sweep knee: 0.99 (floor 1%%) gives OCC precision 0.98 / recall 0.89")
    ap.add_argument("--tolerance", type=int, default=200)              # Ostovar (absolute cases)
    ap.add_argument("--tolerance-frac", type=float, default=0.15)      # Ceravolo (fraction of trace count)
    ap.add_argument("--cer-tol", type=int, default=200)               # Ceravolo absolute window (0 = use frac); 200 = Ostovar parity for the 1000-trace workflow
    ap.add_argument("--sizes", default="", help="Ceravolo size filter, e.g. '1000' (default all)")
    ap.add_argument("--batch-size", type=int, default=0, help="override auto batch size (0 = auto ~100)")
    ap.add_argument("--span", type=int, default=0, help="override detector span/persistence (0 = auto)")
    ap.add_argument("--min-int-frac", type=float, default=0.0, help="relative intensity filter: drop CPs shifting < this fraction of tracked relations (0 = off, recall-first; 0.01 lifts DETECT precision but coarsens labeling windows -- use for detection reporting, not labeling)")
    ap.add_argument("--level", choices=["overview", "DETECT", "INTENSITY", "CONFIRM", "OCC", "COND", "BRANCH", "SKIP", "REORDER", "FREQUENCY", "SUMMARY", "LEAVES", "EXPLORE", "OCCMISS", "OCCSPLIT", "L1"], default="overview")
    ap.add_argument("--explore-codes", default="cf,cd", help="patterns to dump in EXPLORE mode")
    args = ap.parse_args()

    if args.sizes:
        unknown = {s.strip() for s in args.sizes.split(",") if s.strip()} - set(SIZES)
        if unknown:
            ap.error(f"unknown --sizes {sorted(unknown)}; choose from {','.join(SIZES)}")
    channels = {c.strip().lower() for c in args.channels.split(",") if c.strip()}
    if not channels or not channels <= CHANNELS:
        ap.error(f"--channels must be a non-empty subset of {sorted(CHANNELS)}; got {sorted(channels)}")

    if args.level in ("DETECT", "INTENSITY"):
        print(f"{args.level} | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence} "
              f"| batch={args.batch_size or 'auto'} | tol: ostovar={args.tolerance}, ceravolo={args.cer_tol or str(int(args.tolerance_frac*100))+'%'}")
        recs = list(iter_detect(args, channels))
        if not recs:
            return _no_logs()
        (report_detect if args.level == "DETECT" else report_intensity)(recs)
        return 0
    if args.level == "CONFIRM":
        print(f"CONFIRM (full-regime windows + dual gate) | dataset={args.dataset} sizes={args.sizes or 'all'} "
              f"| K_born={args.born_k}")
        records = list(iter_confirm(args, channels))
        if not records:
            return _no_logs()
        report_confirm(records, args.min_int_frac or 0.01)
        return 0

    drift_rows: list[dict] = []         # one entry per scored/missed true drift
    # per pattern: Counter of level (L1..L4) hits, Counter of fingerprint, total scored, det-misses
    levels: dict[str, Counter] = {}
    prints: dict[str, Counter] = {}
    subs: dict[str, Counter] = {}       # L4 sub-lens fingerprint distribution
    totals: dict[str, list[int]] = {}   # [scored, det_miss]

    if args.dataset == "both":
        drifts = itertools.chain(iter_ostovar(args, channels), iter_ceravolo(args, channels))
    elif args.dataset == "ceravolo":
        drifts = iter_ceravolo(args, channels)
    else:
        drifts = iter_ostovar(args, channels)
    for code, sig in drifts:
        levels.setdefault(code, Counter()); prints.setdefault(code, Counter())
        subs.setdefault(code, Counter()); totals.setdefault(code, [0, 0])
        if sig is None:
            totals[code][1] += 1
            drift_rows.append({"code": code, "miss": True})
            continue
        fired = decisions(sig)
        totals[code][0] += 1
        for L in fired:
            levels[code][L] += 1
        prints[code][fingerprint(fired)] += 1
        sub = l4_sublens(sig)
        if "L4" in fired:
            subs[code][fingerprint(sub)] += 1
        occ = occ_shape(sig, 1.0 - args.born_k)
        drift_rows.append({"code": code, "miss": False, "L1": "L1" in fired, "L2": "L2" in fired,
                           "L3": "L3" in fired, "L4": "L4" in fired, "shape": l1_shape(sig),
                           "occ": bool(occ), "occfp": "+".join(sorted(occ)) or "none",
                           "cond": "ECg" in sub, "branch": branch_tc(sig),
                           "skip": skip_sig(sig), "reorder": reorder_sig(sig),
                           "freq": freq_sig(sig), "sig": sig})

    if not drift_rows:
        return _no_logs()

    # ---- focused per-level reports ----
    if args.level == "OCC":
        print(f"OCC probe | dataset={args.dataset} | test={args.test} K={args.confidence}\n")
        report_occ(drift_rows)
        return 0
    if args.level == "COND":
        print(f"COND probe | dataset={args.dataset} | test={args.test} K={args.confidence}\n")
        report_cond(drift_rows)
        return 0
    if args.level == "BRANCH":
        print(f"BRANCH probe | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_branch(drift_rows)
        return 0
    if args.level == "SKIP":
        print(f"SKIP probe | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_skip(drift_rows)
        return 0
    if args.level == "REORDER":
        print(f"REORDER probe | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_reorder(drift_rows)
        return 0
    if args.level == "FREQUENCY":
        print(f"FREQUENCY probe | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_freq(drift_rows)
        return 0
    if args.level == "LEAVES":
        print(f"LEAVES | dataset={args.dataset} sizes={args.sizes or 'all'} | K_born={args.born_k}\n")
        report_leaves(drift_rows, 1.0 - args.born_k)
        return 0
    if args.level == "SUMMARY":
        print(f"SUMMARY | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_summary(drift_rows)
        return 0
    if args.level == "EXPLORE":
        print(f"EXPLORE | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_explore(drift_rows, {c.strip().lower() for c in args.explore_codes.split(",") if c.strip()})
        return 0
    if args.level == "OCCSPLIT":
        print(f"OCCSPLIT | dataset={args.dataset} sizes={args.sizes or 'all'} | K_born={args.born_k}\n")
        report_occsplit(drift_rows, 1.0 - args.born_k)
        return 0
    if args.level == "OCCMISS":
        print(f"OCCMISS | dataset={args.dataset} sizes={args.sizes or 'all'} | test={args.test} K={args.confidence}\n")
        report_occmiss(drift_rows, {c.strip().lower() for c in args.explore_codes.split(",") if c.strip()})
        return 0
    if args.level == "L1":
        tol = f"{int(args.tolerance_frac*100)}%" if args.dataset == "ceravolo" else args.tolerance
        print(f"L1 probe | {args.dataset} | test={args.test} K={args.confidence} tol={tol}\n")
        report_l1(drift_rows)
        return 0

    # ---- overview report ----
    print(f"Top-level decision-tree separability | Ostovar | test={args.test} K={args.confidence} tol={args.tolerance}\n")
    print(f"{'pat':5}{'L1':>5}{'L2':>5}{'L3':>5}{'L4':>5}{'none':>6}{'scored':>8}{'miss':>6}  expected   pure?")
    pure_hits = 0
    pure_tot = 0
    unscoreable: list[str] = []
    for code in sorted(levels):
        lv = levels[code]
        scored, miss = totals[code]
        none = sum(c for fp, c in prints[code].items() if fp == "{none}")
        exp = EXPECTED_LEVEL.get(code, set())
        # "pure" = every scored drift fires a fingerprint that is a subset of expected AND covers it
        ok = 0
        for fp, c in prints[code].items():
            fset = set() if fp == "{none}" else set(fp.strip("{}").split(","))
            if fset and fset == exp:
                ok += c
        # A code with no EXPECTED_LEVEL entry has no ground truth to match, so `ok` is
        # structurally 0 -- counting its drifts in the denominator only depresses the ratio.
        # Its row still prints (expected shows "-"); it just does not score.
        if exp:
            pure_hits += ok; pure_tot += scored
        else:
            unscoreable.append(code)
        mark = "OK" if scored and ok == scored else ("~" if ok else "X")
        print(f"{code:5}{lv['L1']:>5}{lv['L2']:>5}{lv['L3']:>5}{lv['L4']:>5}{none:>6}"
              f"{scored:>8}{miss:>6}  {','.join(sorted(exp)) or '-':9} {mark} {ok}/{scored}")
    print(f"\nExact-fingerprint == expected-bucket: {pure_hits}/{pure_tot}")
    if unscoreable:
        print(f"  (excluded, no expected bucket defined: {', '.join(unscoreable)})")

    print("\nPer-pattern fingerprint distribution (which top-level decisions co-fire):")
    for code in sorted(prints):
        dist = ", ".join(f"{fp}:{c}" for fp, c in prints[code].most_common())
        print(f"  {code:5} {dist}")

    print("\nL4 sub-lens distribution (EC=ExclusiveChoice cross, ORD=order, TOG=co-occur) among L4-firing drifts:")
    print("  -> tests whether ExclusiveChoice separates the conditional family (cf/cm/cre) from cb/sw/fr:")
    for code in sorted(subs):
        if not subs[code]:
            continue
        dist = ", ".join(f"{fp}:{c}" for fp, c in subs[code].most_common())
        print(f"  {code:5} {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
