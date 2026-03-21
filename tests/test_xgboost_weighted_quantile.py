"""Test the XGBoost-compatible quantile helper."""

import numpy as np

from conformal_tights._xgboost_weighted_quantile import _weighted_quantile


def test_weighted_quantile_matches_xgboost_unweighted_quantile() -> None:
    """The unweighted path should match XGBoost's historical interpolation formula."""
    y = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(
        _weighted_quantile(y, [0.1, 0.5, 0.9]), np.array([1.0, 3.0, 8.0], dtype=np.float64)
    )


def test_weighted_quantile_matches_xgboost_weighted_quantile() -> None:
    """The weighted path should match XGBoost's step-function weighted quantile."""
    y = np.array([1.0, 2.0, 4.0, 8.0])
    sample_weight = np.array([1.0, 3.0, 1.0, 5.0])
    np.testing.assert_allclose(
        _weighted_quantile(y, [0.1, 0.5, 0.9], sample_weight=sample_weight),
        np.array([1.0, 4.0, 8.0], dtype=np.float64),
    )
