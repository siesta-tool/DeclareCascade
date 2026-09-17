#!/usr/bin/env python3
"""Generate the evaluation figures for the DeclareCascade paper.

  python3 scripts/make_figures.py --out-dir results/figures
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, rankdata

from evaluate_cdrift import F1_Score, getTP_FP

ROOT = Path(__file__).resolve().parent.parent
PARAM_COLS = ["Window Size", "SW Step Size", "Min Adaptive Window", "Max Adaptive Window",
              "ADWIN Step Size", "P-Value", "Complete-Window Size", "Detection-Window Size",
              "Stable Period", "MRID", "Epsilon"]

# Top Adams baselines by mean F1 across (Bose, Ceravolo, Ostovar) -- see results/cdrift-comparison.txt.
DEFAULT_BASELINES = ["Earth Mover's Distance", "Bose J", "Process Graph Metrics",
                      "Bose WC", "Martjushev ADWIN J"]
OURS_LABEL = "DeclareCascade (ours)"

# Ground-truth pattern-code -> mechanism group (see diagnose_decision_tree.LEAF_MEMBERS).
# "Single-mechanism" = exactly one valid DeclareCascade label; "Ambiguous" = the two
# multi-membership codes (re: INSERTION or REMOVAL; cm: BRANCH or REORDER, resolved by cascade
# precedence); "Inexpressible" = no valid label at all.
PATTERN_GROUPS = {
    "Single-mechanism": {"cb", "cf", "cp", "lp", "sm", "sw", "pm", "cre", "pre", "rp"},
    "Ambiguous": {"re", "cm"},
    "Composite": {"ior", "iro", "oir", "ori", "rio", "roi"},
    "Frequency": {"cd", "fr"},
    "Inexpressible": {"pl", "sre"},
}
GROUP_ORDER = ["Single-mechanism", "Ambiguous", "Composite", "Frequency", "Inexpressible"]

# A square CD diagram has ~1.5in of label gutter per side, so the long benchmark names are
# abbreviated there (the rank axis carries the numbers).
CD_SHORT_NAME = {
    "Earth Mover's Distance": "EMD",
    "Process Graph Metrics": "PGM",
    "Martjushev ADWIN J": "Martj. ADWIN J",
    "Martjushev ADWIN WC": "Martj. ADWIN WC",
    OURS_LABEL: "DeclareCascade",
}

# Figure 1's compact panels give each method ~0.4in of tick width, so the names are cut to
# codes. make_fig1_panels prints the key to stdout for the caption.
FIG1_SHORT_NAME = {
    OURS_LABEL: "Ours",
    "Earth Mover's Distance": "EMD",
    "Bose J": "BJ",
    "Bose WC": "BWC",
    "Process Graph Metrics": "PGM",
    "Martjushev ADWIN J": "MAJ",
    "Martjushev ADWIN WC": "MAWC",
    "Zheng DBSCAN": "ZDB",
    "ProDrift": "PD",
    "LCDD": "LCDD",
}

# Okabe-Ito palette (Okabe & Ito 2008): distinguishable in colour, under the common forms of
# colour-vision deficiency, and by luminance alone in greyscale print.
PALETTE = {
    "blue": "#0072B2",
    "amber": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
}
# figure 1: three metrics -> three distinct hues
COLORS_PRF = {"Precision": PALETTE["blue"], "Recall": PALETTE["amber"], "F1": PALETTE["green"]}
# figure 2: two shades of one hue, because the segments stack into a single quantity
COLORS_CORRECT = {"unambiguous": PALETTE["blue"], "resolved": PALETTE["skyblue"]}
# figure 4: two datasets -> two distinct hues
COLORS_DATASET = {"ostovar": PALETTE["blue"], "ceravolo": PALETTE["vermillion"]}

# Nemenyi critical values q_alpha (Demsar 2006, Table 5), indexed by number of methods k.
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
           10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
           17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855,
           10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159, 16: 3.196,
           17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319},
}


def set_style(font_size: int) -> None:
    """Times-family serif, large type, no chart junk. Prefers real Times New Roman when the
    system has it and otherwise falls back to the metric-compatible URW/Liberation clones."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 3,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "grid.color": "#CFCFCF",
        "grid.linewidth": 0.6,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "pdf.fonttype": 42,          # embed TrueType rather than Type-3, for camera-ready PDFs
        "ps.fonttype": 42,
    })


def _save(fig, out_dir: Path, stem: str, tight: bool = True) -> None:
    """tight=False keeps the requested figsize exactly (use with constrained layout, which has
    already fitted the content); tight=True trims whitespace but can shift the final aspect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches=bbox)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches=bbox)
    plt.close(fig)
    print(f"Wrote {out_dir / stem}.png and .pdf")


# ------------------------------ per-log scoring (shared) ------------------------------

def _prf(detected, known, lag: int) -> tuple[float, float, float]:
    tp, fp = getTP_FP(detected, known, lag=lag)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / len(known) if known else 0.0
    f1 = F1_Score(detected, known, lag=lag, zero_division=0.0)
    return p, r, f1


def our_per_log(lag: int, cdrift_eval_csv: Path) -> pd.DataFrame:
    """Per-log P/R/F1 for our method, recomputed from the stored changepoints."""
    df = pd.read_csv(cdrift_eval_csv)
    recs = []
    for _, row in df.iterrows():
        p, r, f1 = _prf(ast.literal_eval(row["Detected Changepoints"]),
                         ast.literal_eval(row["Actual Changepoints for Log"]), lag)
        recs.append({"Algorithm": OURS_LABEL, "Log Source": row["Log Source"], "Log": row["Log"],
                     "P": p, "R": r, "F1": f1})
    return pd.DataFrame(recs)


def baseline_per_log(datasets: list[str], algorithms: list[str], lag: int,
                      algorithm_results_csv: Path) -> pd.DataFrame:
    """Per-log P/R/F1 for each baseline at its best config per (algorithm, dataset).

    The winning config is picked by the mean of the pre-computed F1-Score column (same recipe as
    compare_to_cdrift.py, so the ranking matches results/cdrift-comparison.txt); P/R/F1 are then
    recomputed from the raw changepoints for that config only, putting them on the same footing
    as our own numbers."""
    df = pd.read_csv(algorithm_results_csv)
    df = df[df["Algorithm"].isin(algorithms) & df["Log Source"].isin(datasets)].copy()
    df["F1_existing"] = pd.to_numeric(df["F1-Score"], errors="coerce").fillna(0.0)

    g = (df.groupby(["Algorithm", "Log Source"] + PARAM_COLS, dropna=False)
           .agg(F1=("F1_existing", "mean")).reset_index())
    best_cfg = g.loc[g.groupby(["Algorithm", "Log Source"])["F1"].idxmax()]

    recs = []
    for _, cfg in best_cfg.iterrows():
        mask = (df["Algorithm"] == cfg["Algorithm"]) & (df["Log Source"] == cfg["Log Source"])
        for c in PARAM_COLS:
            v = cfg[c]
            mask &= df[c].isna() if pd.isna(v) else (df[c] == v)
        for _, row in df[mask].iterrows():
            try:
                det = ast.literal_eval(row["Detected Changepoints"])
                known = ast.literal_eval(row["Actual Changepoints for Log"])
            except (ValueError, SyntaxError):
                continue
            p, r, f1 = _prf(det, known, lag)
            recs.append({"Algorithm": row["Algorithm"], "Log Source": row["Log Source"],
                         "Log": row["Log"], "P": p, "R": r, "F1": f1})
    return pd.DataFrame(recs)


# ------------------------------- Figure 1: detection P/R/F1 -------------------------------

KEYS_PRF = {"Precision": "P", "Recall": "R", "F1": "F1"}

# figures 2-4 are drawn about this wide; figure 1 is much wider, so a page placing them all at the
# same width shrinks figure 1 more. Scaling figure 1's point sizes by its width ratio to this
# reference makes its rendered text come out the same size as the other figures'.
FIG_REF_WIDTH_IN = 9.6


def _prf_panel(ax, per_log: pd.DataFrame, dataset: str, methods: list[str], font_size: int,
               show_ylabel: bool, scale: float = 1.0) -> None:
    means = (per_log[per_log["Log Source"] == dataset]
             .groupby("Algorithm")[["P", "R", "F1"]].mean())
    x = np.arange(len(methods))
    width = 0.27
    for i, metric in enumerate(("Precision", "Recall", "F1")):
        vals = [means.loc[m, KEYS_PRF[metric]] if m in means.index else np.nan for m in methods]
        ax.bar(x + (i - 1) * width, vals, width, label=metric, color=COLORS_PRF[metric],
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([FIG1_SHORT_NAME.get(m, m) for m in methods],
                       fontsize=(font_size - 5) * scale)
    for i, m in enumerate(methods):
        if m == OURS_LABEL:
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    # percentage scale, matching figures 2 and 4
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.tick_params(axis="y", labelsize=(font_size - 4) * scale)
    ax.set_xlabel(dataset, fontsize=(font_size-2) * scale)
    if show_ylabel:
        ax.set_ylabel("Score", fontsize=(font_size-1) * scale)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def make_fig1_panels(per_log: pd.DataFrame, datasets: list[str], methods: list[str], lag: int,
                     out_dir: Path, width_in: float, aspect: float, font_size: int,
                     legend_frac: float, font_scale: float = 1.0) -> None:
    """One compact panel per dataset, sharing a y axis, with the P/R/F1 legend parked in the
    reserved gap between panels so it costs no vertical space."""
    n_panels = len(datasets)
    scale = font_scale * width_in / FIG_REF_WIDTH_IN
    # constrained layout (not tight_layout) copes with the blank legend axes below
    fig = plt.figure(figsize=(width_in, width_in / aspect), layout="constrained")
    # a real (invisible) axes in the middle reserves the legend's width, so the panels cannot
    # grow over it
    ratios = [1.0, legend_frac] + [1.0] * (n_panels - 1) if n_panels > 1 else [1.0]
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.08)

    axes, ax_mid = [], None
    first = fig.add_subplot(gs[0])
    axes.append(first)
    if n_panels > 1:
        ax_mid = fig.add_subplot(gs[1])
        ax_mid.set_axis_off()
        for j in range(1, n_panels):
            axes.append(fig.add_subplot(gs[1 + j], sharey=first))

    for idx, (ax, ds) in enumerate(zip(axes, datasets)):
        _prf_panel(ax, per_log, ds, methods, font_size, show_ylabel=(idx == 0), scale=scale)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    key_handles = [Line2D([], [], linestyle="none") for _ in methods]
    key_labels = [f"{FIG1_SHORT_NAME.get(m, m)} = {'DeclareCascade' if m == OURS_LABEL else m}"
                  for m in methods]

    if ax_mid is not None:
        # both legends stack in the reserved middle gutter: metrics on top, the method-code key
        # under it. add_artist keeps the first legend when the second is attached to the same axes.
        # they anchor past the axes box (top > 1, bottom < 0) so the stack may use the figure's
        # full height, including the strip the panels give up to their x labels.
        leg_metrics = ax_mid.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                                     frameon=False, ncol=1, fontsize=(font_size - 5) * scale,
                                     handlelength=1.1, handletextpad=0.5, labelspacing=0.5,
                                     borderaxespad=0.0)
        ax_mid.add_artist(leg_metrics)
        ax_mid.legend(key_handles, key_labels, loc="lower center", bbox_to_anchor=(0.5, 0),
                       frameon=False, ncol=1, fontsize=(font_size - 5) * scale, handlelength=0.0,
                       handletextpad=0.0, labelspacing=0.4, borderaxespad=0.1, alignment="left")
    else:
        # single panel: no gutter exists, so the key goes below the axes
        axes[0].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                        frameon=False, fontsize=(font_size - 5) * scale, handlelength=1.2)
        fig.legend(key_handles, key_labels, loc="outside lower center", ncol=3, frameon=False,
                   fontsize=(font_size - 7) * scale, handlelength=0.0, handletextpad=0.0,
                   columnspacing=2.0, labelspacing=0.35)

    stem = "fig1_detection_prf" if n_panels > 1 else f"fig1_detection_prf_{datasets[0].lower()}"
    _save(fig, out_dir, stem, tight=False)
    for ds in datasets:
        n_logs = per_log[(per_log["Log Source"] == ds) &
                         (per_log["Algorithm"] == OURS_LABEL)]["Log"].nunique()
        print(f"    ({ds}: {n_logs} log(s), lag={lag} cases)")


# --------------------------- Figure 2: labelling accuracy by group ---------------------------

def make_fig2(label_accuracy_csv: Path, out_dir: Path, width_in: float, aspect: float,
              font_size: int) -> None:
    rows = list(csv.DictReader(label_accuracy_csv.open()))
    covered = {c for codes in PATTERN_GROUPS.values() for c in codes}
    unmapped = {r["pattern_code"] for r in rows} - covered
    if unmapped:
        print(f"[warn] pattern codes not in any group (excluded): {sorted(unmapped)}")

    stats = []
    for g in GROUP_ORDER:
        rs = [r for r in rows if r["pattern_code"] in PATTERN_GROUPS[g]]
        n = len(rs)
        cu = sum(1 for r in rs if r["category"] == "correct_unambiguous")
        cr = sum(1 for r in rs if r["category"] == "correct_resolved")
        stats.append({"group": g, "n": n, "cu": cu, "cr": cr,
                      "accuracy": (cu + cr) / n if n else 0.0})

    # Narrower than figure 3 on purpose: the group names are rotated so they still fit without
    # overlapping, which buys back width at the cost of a bit of extra height (tight bbox grows
    # to fit the rotated labels).
    fig, ax = plt.subplots(figsize=(width_in, width_in / aspect))
    x = np.arange(len(stats))
    unamb = [s["cu"] / s["n"] if s["n"] else 0.0 for s in stats]
    resolved = [s["cr"] / s["n"] if s["n"] else 0.0 for s in stats]

    ax.bar(x, unamb, width=0.6, label="Correct (unambiguous)",
           color=COLORS_CORRECT["unambiguous"], zorder=3)
    ax.bar(x, resolved, width=0.6, bottom=unamb, label="Correct (ambiguity resolved)",
           color=COLORS_CORRECT["resolved"], zorder=3)
    for i, s in enumerate(stats):
        ax.text(i, s["accuracy"] + 0.025, f"{s['accuracy']*100:.0f}% ($n$={s['n']})",
                ha="center", va="bottom", fontsize=font_size - 3)

    ax.set_xticks(x)
    ax.set_xticklabels([s["group"] for s in stats], fontsize=font_size - 1,
                       rotation=20, ha="right")
    ax.set_ylabel("Labelling accuracy", fontsize=font_size - 2)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, borderpad=0.35, fontsize=font_size - 6,
              handlelength=1.2)
    fig.tight_layout(pad=0.3)
    _save(fig, out_dir, "fig2_labelling_by_group")


# ------------------ Figure 4: labelling accuracy by group, split per dataset ------------------

def make_fig4(label_accuracy_csv: Path, out_dir: Path, width_in: float, aspect: float,
              font_size: int) -> None:
    """Same grouping as figure 2, but Ostovar vs Ceravolo side by side -- the pooled view hides
    large per-dataset splits (e.g. the ambiguous group is 100% on Ostovar and 0% on Ceravolo)."""
    rows = list(csv.DictReader(label_accuracy_csv.open()))
    datasets = [ds for ds in ("ostovar", "ceravolo") if any(r["dataset"] == ds for r in rows)]

    fig, ax = plt.subplots(figsize=(width_in, width_in / aspect))
    colors = COLORS_DATASET
    x = np.arange(len(GROUP_ORDER))
    width = 0.8 / max(len(datasets), 1)

    for d, ds in enumerate(datasets):
        accs, ns = [], []
        for g in GROUP_ORDER:
            rs = [r for r in rows if r["dataset"] == ds and r["pattern_code"] in PATTERN_GROUPS[g]]
            ok = sum(1 for r in rs if r["category"] in ("correct_unambiguous", "correct_resolved"))
            accs.append(ok / len(rs) if rs else 0.0)
            ns.append(len(rs))
        offset = (d - (len(datasets) - 1) / 2) * width
        ax.bar(x + offset, accs, width, label=ds.capitalize(), color=colors.get(ds), zorder=3)
        for xi, (a, n) in enumerate(zip(accs, ns)):
            # n=0 means the pattern group simply does not occur in that benchmark
            ax.text(xi + offset, a + 0.015, f"{n}" if n else "0", ha="center", va="bottom",
                    fontsize=font_size - 6)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Labelling accuracy")
    # headroom for the count annotation above a bar that reaches 100% (Ostovar/Ambiguous does)
    ax.set_ylim(0, 1.09)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, borderpad=0.4, title=None)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_labelling_by_group_per_dataset")


# ---------------------- Figure 3: critical-difference diagram (Nemenyi) ----------------------

def nemenyi_cd(k: int, n: int, alpha: float) -> float:
    """Critical difference for the Nemenyi post-hoc test: q_alpha * sqrt(k(k+1)/(6n))."""
    table = NEMENYI_Q.get(alpha)
    if table is None:
        raise SystemExit(f"no Nemenyi table for alpha={alpha}; use one of {sorted(NEMENYI_Q)}")
    if k not in table:
        raise SystemExit(f"Nemenyi table covers k=2..{max(table)}; got k={k}")
    return table[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def rank_matrix(per_log: pd.DataFrame, methods: list[str], datasets: list[str]) -> pd.DataFrame:
    """logs x methods matrix of F1, restricted to logs every method covers."""
    sub = per_log[per_log["Log Source"].isin(datasets) & per_log["Algorithm"].isin(methods)]
    mat = sub.pivot_table(index=["Log Source", "Log"], columns="Algorithm", values="F1")
    mat = mat.dropna(axis=0, how="any")
    return mat[[m for m in methods if m in mat.columns]]


def make_fig3(per_log: pd.DataFrame, methods: list[str], datasets: list[str], scope: str,
              alpha: float, out_dir: Path, font_size: int, width_in: float,
              row_inch: float, gutter_in: float) -> None:
    mat = rank_matrix(per_log, methods, datasets)
    n, k = mat.shape
    if n < 2 or k < 2:
        print(f"[warn] skipping CD diagram '{scope}': need >=2 logs and >=2 methods (got {n}, {k})")
        return

    # rank 1 = best; ranks are per-log, averaged over logs. Ties get the average rank.
    ranks = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, mat.to_numpy())
    avg_ranks = pd.Series(ranks.mean(axis=0), index=mat.columns).sort_values()
    cd = nemenyi_cd(k, n, alpha)
    try:
        stat, p = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    except ValueError:                     # identical columns make the statistic undefined
        stat, p = np.nan, np.nan

    print(f"  CD diagram [{scope}]: n={n} logs, k={k} methods, CD={cd:.2f} (alpha={alpha}), "
          f"Friedman chi2={stat:.1f} p={p:.2e}")

    # ---- cliques: maximal runs of methods within CD of each other ----
    r_vals = avg_ranks.to_numpy()
    spans = []
    for i in range(k):
        j = i
        while j + 1 < k and r_vals[j + 1] - r_vals[i] <= cd:
            j += 1
        if j > i:
            spans.append((i, j))
    spans = [s for s in spans
             if not any(s != t and t[0] <= s[0] and s[1] <= t[1] for t in spans)]

    # ---- layout ----
    # A CD diagram is wide and short, not square. Geometry is driven in INCHES and converted to
    # rank units, so the label gutters always hold the (abbreviated) names no matter how many
    # methods there are: text overflowing xlim would make bbox_inches="tight" grow the canvas and
    # squash the rank axis into a sliver.
    #   y is measured in "rows": 1 unit = one label row = `row_inch` inches.
    names = list(avg_ranks.index)
    lo, hi = 1, k
    span = hi - lo
    per_side = (k + 1) // 2
    label_fs = font_size - 5

    axis_in = max(1.5, width_in - 2 * gutter_in)          # inches available to the rank axis
    gutter = span * gutter_in / axis_in                    # same gutter, in rank units
    elbow = 0.22 * gutter                                  # elbow reaches this far into the gutter

    clique_ys = [-(0.30 + 0.26 * idx) for idx in range(len(spans))]
    first_row = 0.30 + 0.26 * max(len(spans), 1) + 0.55
    # one text line, in y-units, so the rank ticks and the CD ruler never collide regardless of
    # --font-size / --cd-row-inch
    text_h = (label_fs / 72.0) / row_inch
    y_ticks = 0.20                                  # baseline of the rank tick labels
    y_cd = y_ticks + text_h + 0.20                  # CD ruler clears those labels
    y_cd_text = y_cd + 0.12
    y_top, y_bot = y_cd_text + text_h + 0.10, -(first_row + (per_side - 1) + 0.45)
    height_in = (y_top - y_bot) * row_inch

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_axis_off()
    ax.set_xlim(lo - gutter, hi + gutter)
    ax.set_ylim(y_bot, y_top)

    ax.plot([lo, hi], [0, 0], color="black", lw=1.1, zorder=3)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.12], color="black", lw=1.1, zorder=3)
        ax.text(r, y_ticks, str(r), ha="center", va="bottom", fontsize=label_fs)

    left_x, right_x = lo - elbow, hi + elbow
    for i, name in enumerate(names):
        r = avg_ranks[name]
        on_left = i < per_side
        y = -(first_row + (i if on_left else k - 1 - i))
        end = left_x if on_left else right_x
        ax.plot([r, r], [0, y], color="black", lw=0.9, zorder=2)
        ax.plot([r, end], [y, y], color="black", lw=0.9, zorder=2)
        weight = "bold" if name == OURS_LABEL else "normal"
        ax.text(end + (-0.015 if on_left else 0.015) * span, y,
                CD_SHORT_NAME.get(name, name), ha="right" if on_left else "left", va="center",
                fontsize=label_fs, fontweight=weight)

    for (i, j), y in zip(spans, clique_ys):
        ax.plot([r_vals[i] - 0.04, r_vals[j] + 0.04], [y, y], color="black", lw=3.4,
                solid_capstyle="butt", zorder=4)

    # ---- CD ruler, above the rank tick labels ----
    ax.plot([lo, lo + cd], [y_cd, y_cd], color="black", lw=1.2)
    for xe in (lo, lo + cd):
        ax.plot([xe, xe], [y_cd - 0.09, y_cd + 0.09], color="black", lw=1.2)
    ax.text(lo + cd / 2, y_cd_text, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=label_fs)

    fig.tight_layout(pad=0.2)
    _save(fig, out_dir, f"fig3_cd_{scope}")


# ------------------ Figure 5: one-at-a-time detection-parameter sweeps ------------------
# Consumes results/param_sweeps.csv (scripts/param_sweeps.py): mean F1 (top row) and mean
# localisation error (Avg-Lag, bottom row) across four one-at-a-time sweeps -- intensity
# threshold xi, trailing-window width s, batch size |B|, and the localisation policy --
# replacing the old xi-only sensitivity table.

FIG5_DATASETS = ["ceravolo", "ostovar"]          # Bose is one log; no per-dataset mean is meaningful
FIG5_MARKERS = {"ceravolo": "o", "ostovar": "s"}
FIG5_ANCHOR_COLOR = "#8A8A8A"

REPORT_ORDER = ["start", "mid", "end", "adaptive", "refine"]
# "adaptive" is long against a narrow tick pitch; the other four fit as written
REPORT_SHORT = {"adaptive": "adapt."}


def _fig5_xi_label(v: float) -> str:
    """0.005 -> '.005'. xi is always < 1, so the leading zero is pure width with no ambiguity --
    load-bearing at a ~0.2in tick pitch, not cosmetic."""
    return "0" if v == 0 else f"{v:g}".lstrip("0")


def _fig5_int_label(v: float) -> str:
    return f"{int(round(v))}"


# One entry per sweep column, left to right. `units` are RELATIVE panel widths: xi has 8 swept
# values and report has worded labels, so both get more of the row than s/|B| -- equal-width
# panels would crowd the 8 xi ticks into the same space as B's 6.
FIG5_SWEEPS = [
    {"sweep": "xi",     "col": "min_int_frac", "xlabel": r"$\xi$", "label": _fig5_xi_label,  "units": 8},
    {"sweep": "s",      "col": "span",         "xlabel": r"$s$",   "label": _fig5_int_label, "units": 5},
    {"sweep": "B",      "col": "batch_size",   "xlabel": r"$|B|$", "label": _fig5_int_label, "units": 6},
    {"sweep": "report", "col": "report",       "xlabel": "policy", "label": None,            "units": 7},
]


def _fig5_positions(sub: pd.DataFrame, spec: dict, omit: set[str] | None = None) -> list:
    """Ordered swept values for one axis, as strings matching the CSV's `value` column.

    `omit` drops swept values from the FIGURE only -- the sweep CSV still carries them, so the
    omission is a presentation choice and never silently loses data."""
    vals = [v for v in dict.fromkeys(sub["value"].astype(str).tolist()) if v not in (omit or set())]
    if spec["sweep"] == "report":
        return [v for v in REPORT_ORDER if v in vals] + [v for v in vals if v not in REPORT_ORDER]
    return sorted(vals, key=float)


def make_fig5(param_sweeps_csv: Path, out_dir: Path, width_in: float, aspect: float,
              font_size: int, lag_scale: str, min_boundaries: int, min_tp_coverage: float,
              omit_policies: set[str] | None = None) -> None:
    """Mean F1 (top row) and mean Avg-Lag (bottom row) across the four sweeps, one column per
    parameter, Ceravolo vs Ostovar. Two things make a sweep cell not comparable, and they get
    different treatment:

      * too few statistically testable boundaries (`n_boundaries`). A trailing window of s
        batches leaves only the boundaries in [s, n_batches - s] testable, so on Ceravolo (1000
        cases, |B|=100 -> 10 batches) s=5 leaves exactly ONE -- sitting on the true change point
        at case 499, so F1 would be 1.0 by construction, not merit. Such cells are EXCLUDED (the
        line breaks) rather than plotted, since a fabricated F1 of 1.0 is actively misleading
        rather than merely imprecise.
      * too few logs contributed a true positive (`logs_with_tp`). get_avg_lag averages only
        assigned pairs, so a config that stops detecting almost everything reports the lag of
        the few logs it still hits and looks like *better* localisation. F1 is immune (our
        zero_division=0.0 convention scores a no-TP log as 0.0 rather than dropping it), so this
        is flagged with a hollow marker on the LAG row only, not excluded -- the point is real,
        just not comparable to a fully-covered config.
    """
    df = pd.read_csv(param_sweeps_csv)
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["sweep"] = df["sweep"].astype(str).str.strip()
    for col in ("mean_f1", "mean_lag", "n_boundaries", "n_logs", "logs_with_tp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    datasets = [d for d in FIG5_DATASETS if (df["dataset"] == d).any()]
    if not datasets:
        print(f"[warn] skipping figure 5: no Ceravolo/Ostovar rows in {param_sweeps_csv.name}")
        return

    fig, axes = plt.subplots(2, len(FIG5_SWEEPS), figsize=(width_in, width_in / aspect),
                             sharey="row", layout="constrained",
                             gridspec_kw={"width_ratios": [s["units"] for s in FIG5_SWEEPS],
                                          "wspace": 0.06, "hspace": 0.12})

    # scale from the cells that are actually DRAWN: Bose is not plotted and boundary-degenerate
    # cells are excluded, so letting their lags set the range (or force a log->linear fallback)
    # would be driven by data the reader never sees
    drawn = df[df["dataset"].isin(datasets) & (df["n_boundaries"] >= min_boundaries)]
    lag_all = drawn["mean_lag"].dropna()
    lag_lo = float(lag_all.min()) if len(lag_all) else 1.0
    lag_hi = float(lag_all.max()) if len(lag_all) else 100.0
    use_log = lag_scale == "log" and lag_lo > 0
    if lag_scale == "log" and not use_log:
        print("[warn] figure 5: a drawn mean Avg-Lag of 0 is present, so the lag row falls back "
              "to a linear scale")

    excluded_notes, hollow_any = [], False
    for c, spec in enumerate(FIG5_SWEEPS):
        sub = df[df["sweep"] == spec["sweep"]]
        ax_f1, ax_lag = axes[0, c], axes[1, c]
        ax_f1.sharex(ax_lag)
        if sub.empty:
            print(f"[warn] figure 5: no rows for sweep '{spec['sweep']}' in {param_sweeps_csv.name}")
            ax_f1.set_axis_off()
            ax_lag.set_axis_off()
            continue

        order = _fig5_positions(sub, spec, omit_policies)
        x = np.arange(len(order))
        anchor_i = next((i for i, v in enumerate(order)
                         if ((sub["value"].astype(str) == v) & (sub["is_anchor"] == 1)).any()), None)

        for ds in datasets:
            rows_by_val = {str(row["value"]): row for _, row in sub[sub["dataset"] == ds].iterrows()}
            f1 = np.full(len(order), np.nan)
            lag = np.full(len(order), np.nan)
            hollow = np.zeros(len(order), dtype=bool)
            cut_at = None
            for i, v in enumerate(order):
                row = rows_by_val.get(v)
                if row is None:
                    continue
                nb = row["n_boundaries"]
                if pd.notna(nb) and nb < min_boundaries:
                    if cut_at is None:
                        cut_at = i
                    continue                                # excluded: leave NaN, line breaks here
                f1[i], lag[i] = row["mean_f1"], row["mean_lag"]
                n_logs, tp = row["n_logs"], row["logs_with_tp"]
                if pd.notna(n_logs) and n_logs > 0 and pd.notna(tp) and tp / n_logs < min_tp_coverage:
                    hollow[i] = True
                    hollow_any = True

            color, marker = COLORS_DATASET[ds], FIG5_MARKERS[ds]
            for ax, y in ((ax_f1, f1), (ax_lag, lag)):
                ax.plot(x, y, color=color, marker=marker, ms=4.2, lw=1.3, mew=0.0, zorder=4,
                        label=ds.capitalize() if (ax is ax_f1 and c == 0) else None)
            bad = np.where(hollow & ~np.isnan(lag))[0]
            if len(bad):
                ax_lag.plot(x[bad], lag[bad], ls="none", marker=marker, ms=4.2, mfc="white",
                           mec=color, mew=1.1, zorder=5)
            if anchor_i is not None:
                for ax in (ax_f1, ax_lag):
                    ax.axvline(anchor_i, color=FIG5_ANCHOR_COLOR, lw=0.9, ls=(0, (3.5, 2.5)), zorder=1)
            if cut_at is not None:
                excluded_notes.append(f"{spec['sweep']}: {ds} excluded from {order[cut_at]} onward "
                                      f"(< {min_boundaries} testable boundaries)")

        for ax in (ax_f1, ax_lag):
            ax.set_xlim(-0.35, len(order) - 1 + 0.35)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=font_size - 6, length=2.5, width=0.7, pad=1.8)

        labels = ([spec["label"](float(v)) for v in order] if spec["label"]
                 else [REPORT_SHORT.get(v, v) for v in order])
        ax_lag.set_xticks(x)
        ax_lag.set_xticklabels(labels, fontsize=font_size - 6)
        if anchor_i is not None:
            ax_lag.get_xticklabels()[anchor_i].set_fontweight("bold")
        ax_lag.set_xlabel(spec["xlabel"], fontsize=font_size - 4, labelpad=2.0)
        ax_f1.tick_params(axis="x", labelbottom=False)
        if c:
            ax_f1.tick_params(axis="y", labelleft=False)
            ax_lag.tick_params(axis="y", labelleft=False)

    top, low = axes[0, 0], axes[1, 0]
    top.set_ylim(0, 1.05)
    top.set_yticks(np.arange(0, 1.01, 0.25))
    top.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    top.set_ylabel("Mean F1", fontsize=font_size - 5, labelpad=2.0)
    if use_log:
        # log, not linear: report=start (~99 cases) is ~20x the other policies, so a shared
        # linear axis would flatten the xi/s/|B| panels onto the bottom spine.
        low.set_yscale("log")
        low.set_ylim(lag_lo / 1.7, lag_hi * 1.7)
    else:
        low.set_ylim(0, lag_hi * 1.08 if lag_hi > 0 else 1.0)
    low.set_ylabel("Mean Avg-Lag\n(cases)", fontsize=font_size - 5, labelpad=2.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=FIG5_ANCHOR_COLOR, lw=0.9, ls=(0, (3.5, 2.5))))
    labels.append("default config")
    if hollow_any:
        handles.append(Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec="#444444", mew=1.1))
        labels.append("Avg-Lag over few logs")
    fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), frameon=False,
              fontsize=font_size - 5, handlelength=1.6, handletextpad=0.5, columnspacing=1.3,
              borderaxespad=0.0)

    _save(fig, out_dir, "fig5_param_sweeps", tight=False)
    for line in excluded_notes:
        print(f"    [excluded] {line}")
    print("    (x axes are ordinal -- swept values are equally spaced, so segment slope is not a "
          "derivative; state this in the caption)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--cdrift-eval-csv", type=Path, default=ROOT / "results" / "cdrift-eval.csv")
    ap.add_argument("--algorithm-results-csv", type=Path, default=ROOT / "cdrift-evaluation" / "algorithm_results.csv")
    ap.add_argument("--label-accuracy-csv", type=Path, default=ROOT / "results" / "label_accuracy_results.csv")
    ap.add_argument("--datasets", default="Ceravolo,Ostovar,Bose",
                     help="datasets to plot in figure 1 (one square figure each)")
    ap.add_argument("--baselines", default=",".join(DEFAULT_BASELINES),
                     help="comma-separated Adams algorithm names for figure 1")
    ap.add_argument("--cd-baselines", default="all",
                     help="'all' (every Adams method) or a comma-separated subset, for figure 3")
    ap.add_argument("--lag", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05, choices=sorted(NEMENYI_Q))
    ap.add_argument("--font-size", type=int, default=18)
    ap.add_argument("--figsize", type=float, default=8.6,
                     help="figure 4 width in inches")
    ap.add_argument("--fig4-aspect", type=float, default=1.95,
                     help="figure 4 width:height (1.0 = square)")
    ap.add_argument("--cd-width", type=float, default=9.6,
                     help="figure 3 width in inches (CD diagrams are wide and short, not square)")
    ap.add_argument("--cd-row-inch", type=float, default=0.26,
                     help="figure 3 vertical inches per label row -- lower = more compact")
    ap.add_argument("--cd-gutter-in", type=float, default=1.35,
                     help="figure 3 inches reserved per side for the method labels")
    ap.add_argument("--fig2-width", type=float, default=8.64,
                     help="figure 2 width in inches (independent of --cd-width: the rotated "
                          "group labels let figure 2 run narrower than figure 3)")
    ap.add_argument("--fig2-aspect", type=float, default=3.0,
                     help="figure 2 width:height before the tight bbox grows it to fit the "
                          "rotated labels")
    ap.add_argument("--fig1-width", type=float, default=15.0,
                     help="figure 1 width in inches (wider = wider panels)")
    ap.add_argument("--fig1-aspect", type=float, default=4.4,
                     help="figure 1 width:height (paneled, compact)")
    ap.add_argument("--fig1-single-aspect", type=float, default=3.0,
                     help="figure 1 width:height for a single-dataset panel; taller than the "
                          "paneled default because with no middle gutter both legends have to "
                          "stack above and below the axes")
    ap.add_argument("--fig1-font-scale", type=float, default=0.72,
                     help="figure 1 type size relative to figures 2-4 once the width difference "
                          "is accounted for; 1.0 renders it at the same size on the page")
    ap.add_argument("--fig1-legend-frac", type=float, default=0.42,
                     help="figure 1 legend gutter width as a fraction of one panel's width; it "
                          "holds both the metric legend and the method-code key")
    ap.add_argument("--param-sweeps-csv", type=Path, default=ROOT / "results" / "param_sweeps.csv")
    ap.add_argument("--fig5-width", type=float, default=9.6, help="figure 5 width in inches")
    ap.add_argument("--fig5-aspect", type=float, default=2.6,
                     help="figure 5 width:height (two rows of four panels)")
    ap.add_argument("--fig5-lag-scale", choices=["log", "linear"], default="log",
                     help="figure 5 bottom row; log keeps the flat xi/s/|B| sweeps readable next "
                          "to report=start, which is ~20x larger")
    ap.add_argument("--fig5-min-boundaries", type=int, default=4,
                     help="figure 5 excludes (breaks the line for) configs with fewer than this "
                          "many statistically testable boundaries -- their F1 is not meaningful")
    ap.add_argument("--fig5-min-tp-coverage", type=float, default=0.6,
                     help="figure 5 hollows the LAG marker when fewer than this fraction of logs "
                          "produced a true positive, since Avg-Lag averages only those")
    ap.add_argument("--fig5-omit-policies", default="refine",
                     help="localisation policies to leave OUT of figure 5's policy panel "
                          "(presentation only -- results/param_sweeps.csv still carries them). "
                          "Pass an empty string to plot all five.")
    ap.add_argument("--only", choices=["1", "2", "3", "4", "5", "all"], default="all")
    args = ap.parse_args()

    set_style(args.font_size)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    algorithms = [a.strip() for a in args.baselines.split(",") if a.strip()]

    per_log = None
    if args.only in ("1", "3", "all"):
        if not args.cdrift_eval_csv.exists():
            raise SystemExit(f"missing {args.cdrift_eval_csv} -- run evaluate_cdrift.py first")
        if not args.algorithm_results_csv.exists():
            raise SystemExit(f"missing {args.algorithm_results_csv}")
        all_algos = sorted(pd.read_csv(args.algorithm_results_csv, usecols=["Algorithm"])["Algorithm"].unique())
        cd_algos = all_algos if args.cd_baselines == "all" else \
            [a.strip() for a in args.cd_baselines.split(",") if a.strip()]
        wanted = sorted(set(algorithms) | set(cd_algos))
        print(f"Scoring per-log P/R/F1 (lag={args.lag}) for {len(wanted)} baselines + ours ...")
        per_log = pd.concat([our_per_log(args.lag, args.cdrift_eval_csv),
                             baseline_per_log(datasets, wanted, args.lag, args.algorithm_results_csv)],
                            ignore_index=True)

    if args.only in ("1", "all"):
        fig1_methods = [OURS_LABEL] + algorithms
        def n_logs(ds):
            return per_log[(per_log["Log Source"] == ds) &
                           (per_log["Algorithm"] == OURS_LABEL)]["Log"].nunique()
        # single-log datasets (Bose) get their own small figure: one log is not a mean, so it
        # does not belong in the same paneled comparison
        paneled = [ds for ds in datasets if n_logs(ds) > 1]
        if paneled:
            make_fig1_panels(per_log, paneled, fig1_methods, args.lag, args.out_dir,
                             args.fig1_width, args.fig1_aspect, args.font_size,
                             args.fig1_legend_frac, args.fig1_font_scale)
        for ds in (ds for ds in datasets if n_logs(ds) <= 1):
            make_fig1_panels(per_log, [ds], fig1_methods, args.lag, args.out_dir,
                             args.fig1_width, args.fig1_single_aspect, args.font_size,
                             args.fig1_legend_frac, args.fig1_font_scale)

    if args.only in ("2", "4", "all"):
        if not args.label_accuracy_csv.exists():
            raise SystemExit(f"missing {args.label_accuracy_csv} -- run label_accuracy.py first")
        if args.only in ("2", "all"):
            make_fig2(args.label_accuracy_csv, args.out_dir, args.fig2_width, args.fig2_aspect,
                      args.font_size)
        if args.only in ("4", "all"):
            make_fig4(args.label_accuracy_csv, args.out_dir, args.figsize, args.fig4_aspect,
                      args.font_size)

    if args.only in ("3", "all"):
        cd_methods = [OURS_LABEL] + (
            sorted(set(per_log["Algorithm"]) - {OURS_LABEL}) if args.cd_baselines == "all"
            else [a.strip() for a in args.cd_baselines.split(",") if a.strip()])
        # a single-log dataset (Bose) carries no rank information, so pool only the multi-log ones
        multi = [ds for ds in datasets
                 if per_log[(per_log["Log Source"] == ds) &
                            (per_log["Algorithm"] == OURS_LABEL)]["Log"].nunique() > 1]
        cd_geom = dict(font_size=args.font_size, width_in=args.cd_width,
                       row_inch=args.cd_row_inch, gutter_in=args.cd_gutter_in)
        for ds in multi:
            make_fig3(per_log, cd_methods, [ds], ds.lower(), args.alpha, args.out_dir, **cd_geom)
        if len(multi) > 1:
            make_fig3(per_log, cd_methods, multi, "pooled", args.alpha, args.out_dir, **cd_geom)

    if args.only in ("5", "all"):
        # don't hard-fail `--only all` on a CSV a sibling script hasn't produced yet
        if not args.param_sweeps_csv.exists():
            msg = f"missing {args.param_sweeps_csv} -- run param_sweeps.py first"
            if args.only == "5":
                raise SystemExit(msg)
            print(f"[warn] skipping figure 5: {msg}")
        else:
            make_fig5(args.param_sweeps_csv, args.out_dir, args.fig5_width, args.fig5_aspect,
                      args.font_size, args.fig5_lag_scale, args.fig5_min_boundaries,
                      args.fig5_min_tp_coverage,
                      {p.strip() for p in args.fig5_omit_policies.split(",") if p.strip()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
