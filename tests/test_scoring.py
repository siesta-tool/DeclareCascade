"""Fidelity tests for the cdrift scoring copied into scripts/evaluate_cdrift.py.

The four functions are a verbatim copy of `cdrift-evaluation/cdrift/evaluation.py`, and the
published F1 / Average-Lag numbers only mean what the README says they mean if the copy still
agrees with the original. Every fixture below is taken from upstream's own docstrings, so these
tests fail the moment the copy drifts from the benchmark it claims to reproduce.

`cdrift-evaluation/` is not vendored here, so the tests import the copy, never upstream.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from evaluate_cdrift import F1_Score, assign_changepoints, get_avg_lag, getTP_FP

# upstream docstring fixture, used by assign_changepoints / get_avg_lag
DETECTED = [1050, 934, 2100]
ACTUAL = [1000, 1149, 2000]


def test_module_imports_on_current_numpy():
    """`np.NaN` was removed in NumPy 2.0 and was used as a default argument here, so importing the
    module raised AttributeError on a fresh install. Pin the alias out of the source."""
    src = Path(sys.modules[F1_Score.__module__].__file__).read_text()
    assert "np.NaN" not in src
    assert math.isnan(F1_Score([], [1000], 200))          # default zero_division is a real NaN


def test_getTP_FP_upstream_docstring_fixture():
    assert getTP_FP([1000, 1001, 2000], [1000, 2000], 200) == (2, 1)


def test_getTP_FP_without_duplicate_detections():
    # 1001 is inside 1000's lag window, so it is not counted as a false positive
    assert getTP_FP([1000, 1001, 2000], [1000, 2000], 200, False) == (2, 0)


def test_assign_changepoints_upstream_docstring_fixture():
    assert assign_changepoints(DETECTED, ACTUAL, 200) == [(1050, 1149), (934, 1000), (2100, 2000)]


def test_get_avg_lag_upstream_docstring_fixture():
    assert get_avg_lag(DETECTED, ACTUAL, lag=200) == pytest.approx(88.33333333333333)


def test_get_avg_lag_undefined_without_assignments():
    assert math.isnan(get_avg_lag([5000], [1000], lag=200))


@pytest.mark.parametrize(
    "detected, expected",
    [
        ([1199], (1, 0)),   # distance 199: inside
        ([1200], (1, 0)),   # distance 200: the window is inclusive at exactly `lag`
        ([1201], (0, 1)),   # distance 201: outside
    ],
)
def test_lag_window_is_inclusive_at_the_boundary(detected, expected):
    assert getTP_FP(detected, [1000], 200) == expected


@pytest.mark.parametrize("distance, tp", [(199, 1), (200, 1), (201, 0)])
def test_lag_window_boundary_is_symmetric(distance, tp):
    assert getTP_FP([1000 + distance], [1000], 200)[0] == tp
    assert getTP_FP([1000 - distance], [1000], 200)[0] == tp


def test_f1_is_one_when_every_actual_is_matched_exactly():
    assert F1_Score([1000, 2000], [1000, 2000], 200) == pytest.approx(1.0)


def test_f1_penalises_a_false_positive():
    # TP=2, FP=1 -> precision 2/3, recall 1.0
    assert F1_Score([1000, 1001, 2000], [1000, 2000], 200) == pytest.approx(0.8)


def test_f1_zero_division_is_the_callers_choice():
    """evaluate_cdrift.py passes 0.0 rather than upstream's NaN, so logs with no true positive
    score 0 instead of being dropped. That convention is what the published means are computed
    under; changing it changes every published number."""
    assert F1_Score([5000], [1000], 200, zero_division=0.0) == 0.0
    assert math.isnan(F1_Score([5000], [1000], 200))
