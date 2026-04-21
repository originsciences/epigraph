"""Tests for epigraph.common.stats -- FDR correction utilities."""

from __future__ import annotations

import numpy as np

from epigraph.common.stats import apply_fdr_correction


def test_apply_fdr_correction_basic() -> None:
    """Verify q_values and reject arrays have correct shapes and semantics."""
    pvals = np.array([0.01, 0.04, 0.5, 0.8])
    q_values, reject = apply_fdr_correction(pvals, alpha=0.05)

    assert q_values.shape == pvals.shape
    assert reject.shape == pvals.shape
    # q-values should be >= the original p-values (monotonicity of BH)
    assert np.all(q_values >= pvals)
    # reject should be boolean
    assert reject.dtype == bool


def test_apply_fdr_correction_with_nans() -> None:
    """Verify NaN p-values produce NaN q-values and False rejection."""
    pvals = np.array([0.01, np.nan, 0.04, np.nan, 0.5])
    q_values, reject = apply_fdr_correction(pvals, alpha=0.05)

    # NaN positions should stay NaN in q_values
    assert np.isnan(q_values[1])
    assert np.isnan(q_values[3])
    # NaN positions should be False in reject
    assert not reject[1]
    assert not reject[3]
    # Non-NaN positions should have finite q-values
    assert np.isfinite(q_values[0])
    assert np.isfinite(q_values[2])
    assert np.isfinite(q_values[4])


def test_apply_fdr_correction_all_significant() -> None:
    """Verify all True when p-values are tiny."""
    pvals = np.array([1e-10, 1e-8, 1e-6, 1e-5])
    q_values, reject = apply_fdr_correction(pvals, alpha=0.05)

    assert np.all(reject), f"Expected all significant, got reject={reject}"
    assert np.all(q_values < 0.05)


def test_apply_fdr_correction_all_nonsignificant() -> None:
    """Verify all False when p-values are 1.0."""
    pvals = np.array([1.0, 1.0, 1.0, 1.0])
    q_values, reject = apply_fdr_correction(pvals, alpha=0.05)

    assert not np.any(reject), f"Expected none significant, got reject={reject}"
    assert np.all(q_values >= 0.05)
