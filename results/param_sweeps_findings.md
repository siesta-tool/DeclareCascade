# Parameter-sweep findings (ξ, s, |B|, localisation policy)

Source data: `results/param_sweeps.csv` (+ per-log dump `results/param_sweeps_rows.csv`).
Figure: `results/figures/fig5_param_sweeps.pdf`, intended label `fig:param-sweeps`.

## How to read the numbers

- **Anchor / default config**, held fixed while each parameter is swept: ϑ=0.999, ξ=0.01, s=1, |B|=100, policy=end. It reproduces the paper's published Ceravolo 0.911 / Ostovar 0.922 F1 and 5.7 / 20.7 lag exactly.
- **F1**: macro-mean of per-log F1, higher is better.
- **Avg-Lag**: mean |detected − actual| in cases, **lower is better**. It averages *only* logs with ≥1 true positive, so a config that stops detecting shows a flatteringly small lag. Coverage (`logs_with_tp`/`n_logs`) is therefore quoted alongside it and must not be dropped.
- **Testable boundaries** = ⌈N/|B|⌉ − 2s + 1. Cells with <4 are not interpretable and are excluded from the figure (Ceravolo at s≥4 and at |B|≥300).
- Bose is 1 log; it is in the CSV but excluded from all per-dataset conclusions below.

---

## 1. ξ — intensity threshold

| ξ | Cer F1 | Cer lag | Cer cov. | Ost F1 | Ost lag | Ost cov. |
|---|---|---|---|---|---|---|
| 0 | 0.911 | 5.7 | 70/75 | 0.505 | 18.0 | 75/75 |
| 0.005 | 0.911 | 5.7 | 70/75 | 0.835 | 19.3 | 75/75 |
| **0.01** | **0.911** | **5.7** | 70/75 | **0.922** | **20.7** | 75/75 |
| 0.02 | 0.911 | 5.7 | 70/75 | 0.862 | 14.7 | 68/75 |
| 0.03 | 0.844 | 6.2 | 65/75 | 0.779 | 12.5 | 60/75 |
| 0.05 | 0.822 | 4.8 | 63/75 | 0.640 | 12.0 | 50/75 |
| 0.07 | 0.751 | 5.3 | 57/75 | 0.520 | 7.5 | **40/75** |
| 0.10 | 0.662 | 6.0 | 50/75 | 0.333 | 1.9 | **27/75** |

**Outcomes**

- ξ=0.01 is the joint optimum. Ostovar has a sharp peak there (0.505 → 0.922 → 0.333); Ceravolo is flat at 0.911 across ξ∈[0, 0.02] then decays.
- F1 is stable within ξ∈[0.005, 0.02] on both datasets.
- Ceravolo lag is essentially invariant to ξ (4.8–6.2 across the whole range): ξ gates *which* events are confirmed, not where they are placed.

**Paper impact**

- **Confirms** §5.3 and `tab:xi-sweep`: all six values in the existing table reproduce exactly (0.911/0.505/5.7/18.0 … 0.662/0.333/6.0/1.9). No correction needed to those numbers.
- **Correction needed — `tab:xi-sweep` lag column is misleading at high ξ.** The table reports Ostovar lag 1.9 at ξ=0.10 with no indication that it is averaged over **27 of 75 logs**. Read naively it says localisation *improves* as ξ rises, when in fact the detector has stopped finding the hard drifts. Any surviving mention of high-ξ lag must carry the coverage, or the claim should be restricted to F1.

---

## 2. s — trailing-window width

| s | Cer F1 | Cer lag | Cer bounds | Ost F1 | Ost lag | Ost bounds |
|---|---|---|---|---|---|---|
| **1** | **0.911** | **5.7** | 9 | **0.922** | **20.7** | 29 |
| 2 | 0.642 | 92.9 | 7 | 0.624 | 65.1 | 27 |
| 3 | 0.604 | 98.5 | 5 | 0.638 | 71.9 | 25 |
| 4 | *0.867* | *0.0* | **3** | 0.658 | 96.6 | 23 |
| 5 | *0.867* | *100.0* | **1** | 0.688 | 102.1 | 21 |

*Italic Ceravolo cells at s≥4 are degenerate (≤3 testable boundaries) and are excluded from the figure — at s=5 the single testable boundary sits on the true change point, so its F1 is an artifact, not a result.*

**Outcomes**

- s=1 is strictly best and by a wide margin. s=2 alone costs 0.27 F1 on Ceravolo (0.911→0.642) and 0.30 on Ostovar (0.922→0.624), while inflating lag ~16× (5.7→92.9) and ~3× (20.7→65.1).
- Ostovar, which stays statistically comfortable (21–27 boundaries at every s), shows F1 depressed to 0.62–0.69 for *all* s>1. The penalty is therefore real, not an artifact of running out of boundaries.
- Mechanism: a wider trailing window re-tests inside the same transition and re-fires on a drift already reported, and it shrinks the testable-boundary set.

**Paper impact**

- **Confirms** §4.2's default of s=1 in the regime the benchmarks occupy.
- **Caveat to add (untested regime, not a contradiction).** The sweep covers 10-batch (Ceravolo) and 30-batch (Ostovar) streams. `auto_config` selects s=2 *only* when a stream has ≤6 batches — a regime this sweep does not cover. The finding "s=2 is severely harmful" therefore cannot be transferred to that branch as evidence either for or against it; do not claim the sweep validates the ≤6-batch rule.

---

## 3. |B| — batch size

| \|B\| | Cer F1 | Cer lag | Cer cov. | Ost F1 | Ost lag | Ost cov. | aligned? |
|---|---|---|---|---|---|---|---|
| 25 | 0.756 | 38.6 | 68/75 | 0.597 | 38.0 | 69/75 | yes |
| 50 | 0.831 | **3.0** | 66/75 | 0.798 | **16.1** | 73/75 | yes |
| **100** | **0.911** | 5.7 | 70/75 | 0.922 | 20.7 | 75/75 | yes |
| 150 | 0.902 | 104.4 | 68/75 | 0.926 | 78.1 | 72/75 | no |
| 200 | 0.800 | 100.0 | 60/75 | **0.961** | 162.7 | 75/75 | Ost only |
| 300 | *0.493* | *100.0* | *37/75* | 0.847 | 163.0 | 73/75 | no |

*Italic Ceravolo |B|=300 is degenerate (3 testable boundaries) and excluded from the figure.*
*"aligned" = the true change point coincides with a batch's inclusive end (see §4).*

**Outcomes**

- F1 optimum is dataset-dependent: Ceravolo peaks at |B|=100 (0.911), Ostovar at |B|=200 (0.961, vs 0.922 at 100). |B|=100 is the best joint compromise, not the per-dataset optimum.
- Lag optimum is |B|=50 on **both** datasets (3.0 and 16.1), i.e. better than the default.
- |B|=25 degrades **both** metrics (F1 0.756/0.597, lag 38.6/38.0): 25-case batches make per-batch proportions too noisy and the detector fires early and spuriously.
- Lag above the default is driven mainly by grid alignment rather than by |B| itself (see §4). Alignment is necessary but not sufficient: Ostovar at |B|=200 is aligned yet still has lag 162.7, because with coarse batches a one-batch error costs a full |B| cases.

**Paper impact**

- **Contradicts §4.3 as written.** The paper states: *"All four strategies are limited by the batch granularity: smaller batches would improve localization but reduce the sample sizes supporting the significance test."* The trade-off holds only for one step down (100→50: lag 5.7→3.0 and 16.1, F1 0.911→0.831 and 0.922→0.798). Going further to |B|=25 makes **both** worse, so "smaller batches would improve localization" is false as a general statement. Reword to something like: *modestly smaller batches trade F1 for localisation, but below ~50 cases the per-batch proportions become too noisy and both degrade.*
- **Supports** §4.1's |B|=100 default as a joint compromise — but note it is not the lag optimum, which is 50.

---

## 4. Localisation policy

| policy | Cer F1 | Cer lag | Ost F1 | Ost lag |
|---|---|---|---|---|
| start | 0.911 | 99.1 | 0.913 | 87.8 |
| mid | 0.911 | 54.3 | 0.913 | 53.4 |
| **end** | **0.911** | **5.7** | **0.922** | **20.7** |
| adaptive | 0.911 | 60.9 | 0.921 | 78.1 |
| refine | **0.889** | 68.2 | 0.913 | 65.8 |

**Outcomes**

- F1 is essentially invariant to the policy — exactly 0.911 on Ceravolo for start/mid/end/adaptive, and within 0.913–0.922 on Ostovar — because the policy changes only *where inside the firing batch* a change is reported, never *which* batches fire. Lag meanwhile spans 5.7 → 99.1, a ~17× range bounded by |B|. (The residual Ostovar variation is second-order: moving the reported index can push a detection across the 200-case matching window, which is the same mechanism that costs `refine` its F1, just milder.)
- `end` is best on both datasets, ~10× lower lag than the runner-up (`mid`).
- **`end`'s margin is a grid-alignment artifact, not policy quality.** Two benchmark properties coincide: (i) drifts sit at exact multiples of |B| (Ceravolo 1000 cases, drift at the midpoint; Ostovar CPs at 999/1999 with |B|=100), and (ii) ground truth labels the change point as the *last pre-drift case* (n//2 − 1 = 499), which is exactly a batch's inclusive `end_case`. Traced on one log: the detector fires on batches 4 and 5, clustering keeps the peak (batch 4 = cases 400–499), and `end_case` = 499 = the true CP, giving lag 0 by construction.
  - Evidence from the |B| sweep (Ceravolo, policy=end): when 500 is divisible by |B| the lag is 38.6 / 3.0 / 5.7 (|B| = 25 / 50 / 100); when it is not, lag jumps to 104.4 / 100.0 / 100.0 (|B| = 150 / 200 / 300).
  - Decisive test: re-running all five policies at |B|=150 (drift falls mid-batch) **inverts the ranking** — `mid` becomes best (lag 25.0) and `end` becomes worst (104.4).
- `adaptive` underperforms the fixed `end` (60.9 vs 5.7 on Ceravolo). Its Manhattan-distance heuristic picks the wrong edge more often than not; a fixed choice beats it.
- `refine` is the **only** policy that changes F1 (0.889 vs 0.911 on Ceravolo, 0.913 vs 0.922 on Ostovar): it moves reported indices far enough that some detections fall outside the 200-case matching window and are lost.

**Paper impact**

- **Caveat required on the §5.1 headline claim.** *"Mean localisation error is also lowest among the benchmarked methods: 5.7~cases on Ceravolo and 20.7~on Ostovar."* These numbers are real but depend on policy=end being grid-aligned with the benchmarks' synthetic drift placement. State the dependence, or a reviewer who notices the round-number drift placement will read the comparison as tuned. The robust version of the claim is the architectural one: the policy is applied after confirmation, so it moves the reported index within one batch and leaves detection itself untouched (verified: F1 exactly 0.911 on Ceravolo for four of the five policies).
- **Contradicts §4.3's implied benefit of `refine`.** The paper says refinement *"retains the coarse batches required for statistically reliable detection while estimating the change point at a finer temporal resolution."* On these benchmarks refine gives **worse** localisation than plain `end` (68.2 vs 5.7; 65.8 vs 20.7), and is the only policy that *costs* F1. Either drop the implied-improvement framing or state that its sub-batch scan does not beat an edge that the benchmark's convention already sits on.
- **Contradicts the implied benefit of `adaptive`** in §4.3, which presents it as choosing the better edge per event. Measured, it lands between `start` and `end` (60.9 / 78.1) and is beaten by simply always reporting `end`.

---

## 5. Net changes implied for the paper

1. Replace `tab:xi-sweep` with `fig:param-sweeps` (ξ, s, |B|, policy in one figure) and rewrite the §5.3 opening paragraph, which currently covers ξ only. Draft LaTeX: `results/param_sweeps_snippet.tex`.
2. Reword §4.3's "smaller batches would improve localization" — false below |B|≈50.
3. Reword §4.3's `refine` and `adaptive` descriptions to stop implying a localisation benefit neither delivers here.
4. Add the grid-alignment caveat wherever the 5.7 / 20.7 lag figures are quoted as a headline result (§5.1, and the abstract if it repeats them).
5. Wherever high-ξ lag is quoted, carry the coverage (ξ=0.10 Ostovar lag 1.9 is over 27/75 logs).
6. Keep §4.2's s=1 default; do not claim the sweep validates the ≤6-batch s=2 branch, which it does not cover.
