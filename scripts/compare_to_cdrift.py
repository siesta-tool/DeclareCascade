#!/usr/bin/env python3
"""Compare our DeclareTree (results/cdrift-eval.csv) against the cdrift benchmark methods
(cdrift-evaluation/algorithm_results.csv): best-config mean F1 + Average Lag per dataset, and
per-log F1 correlation on Ceravolo. Run after evaluate_cdrift.py; paths resolve to the repo root.

The ranking printed here is NOT the ranking published by Adams et al., and the numbers should not
be quoted as theirs. Four things differ:

  * we take the macro mean of each method's per-log `F1-Score` column; the paper pools TP/FP across
    logs and reports a micro F1, never averaging that column;
  * we score all 151 evaluation logs; the paper's headline table uses a 41-log noiseless subset;
  * we pick the best configuration per (algorithm, dataset); the paper picks one per algorithm, so
    ours is a per-dataset oracle;
  * the grouping columns are ours, not theirs.

On the paper's own subset the gap is large -- e.g. Zheng DBSCAN Ostovar 0.320 -> 0.960, LCDD
0.503 -> 0.619, ProDrift 0.089 -> 0.267. What keeps the comparison usable is that all 151 logs are
scored identically for every method including ours, and that the per-dataset oracle makes the
baselines stronger, not weaker -- so our placing is if anything understated. Treat this as a
like-for-like re-derivation under one consistent recipe, not as a reproduction of the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARAM_COLS = ['Window Size', 'SW Step Size', 'Min Adaptive Window', 'Max Adaptive Window',
              'ADWIN Step Size', 'P-Value', 'Complete-Window Size', 'Detection-Window Size',
              'Stable Period', 'MRID', 'Epsilon']


def main() -> int:
    df = pd.read_csv(ROOT / "cdrift-evaluation" / "algorithm_results.csv")
    missing = [c for c in PARAM_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"algorithm_results.csv missing parameter columns: {missing}")
    param_cols = list(PARAM_COLS)
    df['F1'] = pd.to_numeric(df['F1-Score'], errors='coerce').fillna(0.0)
    df['Lag'] = pd.to_numeric(df['Average Lag'], errors='coerce')

    # Macro mean of the per-log F1 column, best config per (algorithm, dataset) -- our recipe, not
    # the paper's pooled micro-F1 over a single per-algorithm config. See the module docstring.
    g = (df.groupby(['Algorithm', 'Log Source'] + param_cols, dropna=False)
           .agg(F1=('F1', 'mean'), Lag=('Lag', 'mean'), n=('Log', 'nunique')).reset_index())
    best = g.loc[g.groupby(['Algorithm', 'Log Source'])['F1'].idxmax()]
    assert best.groupby('Log Source')['n'].nunique().eq(1).all(), "configs cover differing log sets"

    ours = pd.read_csv(ROOT / "results" / "cdrift-eval.csv")
    ours['F1'] = pd.to_numeric(ours['F1-Score'], errors='coerce').fillna(0.0)
    ours['Lag'] = pd.to_numeric(ours['Average Lag'], errors='coerce')
    our_f1 = ours.groupby('Log Source')['F1'].mean()
    our_lag = ours.groupby('Log Source')['Lag'].mean()

    order = ['Bose', 'Ceravolo', 'Ostovar']
    f1_tab = best.pivot_table(index='Algorithm', columns='Log Source', values='F1')
    lag_tab = best.pivot_table(index='Algorithm', columns='Log Source', values='Lag')
    f1_tab.loc['DeclareTree (ours)'] = our_f1.reindex(order)
    lag_tab.loc['DeclareTree (ours)'] = our_lag.reindex(order)
    f1_tab, lag_tab = f1_tab[order], lag_tab[order]
    # skipna=False: a row missing a dataset must not out-rank a complete one by averaging over fewer
    f1_tab['mean'] = f1_tab.mean(axis=1, skipna=False)
    print("=== best-config mean F1 (higher better) ===")
    print(f1_tab.sort_values('mean', ascending=False).round(3).to_string())
    print("\n=== mean Average Lag at best-F1 config (lower better, cases) ===")
    print(lag_tab.round(1).to_string())

    # per-log F1 correlation on Ceravolo: do we succeed/fail on the same logs as each method?
    print("\n=== Ceravolo per-log F1 correlation with DeclareTree (Pearson r) ===")
    our_cer = ours[ours['Log Source'] == 'Ceravolo'].set_index('Log')['F1']
    for algo in best[best['Log Source'] == 'Ceravolo']['Algorithm']:
        bc = best[(best['Algorithm'] == algo) & (best['Log Source'] == 'Ceravolo')].iloc[0]
        mask = (df['Algorithm'] == algo) & (df['Log Source'] == 'Ceravolo')
        for c in param_cols:
            v = bc[c]
            # safe: v came from a groupby key over this same column, so bitwise-identical
            mask &= (df[c].isna() if pd.isna(v) else df[c] == v)
        s = df[mask].set_index('Log')['F1']
        j = pd.concat([our_cer.rename('ours'), s.rename('them')], axis=1).dropna()
        r = j['ours'].corr(j['them']) if len(j) > 2 and j['them'].std() > 0 and j['ours'].std() > 0 else np.nan
        print(f"  {algo:24} r={r:+.2f}  (n={len(j)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
