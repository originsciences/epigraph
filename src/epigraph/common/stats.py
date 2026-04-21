"""Shared statistical utilities for the epigraph package.

Centralises common patterns such as FDR correction with NaN handling so that
every analysis module applies them identically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests


def apply_fdr_correction(
    pvals: NDArray[np.floating],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Apply FDR correction handling NaN p-values.

    NaN p-values are temporarily replaced with ``1.0`` for the correction
    step and then restored to ``NaN`` in the output.

    Args:
        pvals: 1-D array of raw p-values (may contain ``NaN``).
        method: Correction method passed to
            ``statsmodels.stats.multitest.multipletests``.
        alpha: Family-wise error rate or FDR level.

    Returns:
        Tuple of ``(q_values, reject)`` where *q_values* is the corrected
        p-value array (``NaN`` where the input was ``NaN``) and *reject* is
        a boolean array (``False`` where the input was ``NaN``).
    """
    nan_mask = np.isnan(pvals)
    pvals_clean = np.where(nan_mask, 1.0, pvals)

    reject, qvals, _, _ = multipletests(pvals_clean, alpha=alpha, method=method)

    qvals_final: NDArray[np.floating] = np.where(nan_mask, np.nan, qvals)
    reject_final: NDArray[np.bool_] = np.where(nan_mask, False, reject)

    return qvals_final, reject_final
