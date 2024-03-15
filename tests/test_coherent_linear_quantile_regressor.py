"""Test the Coherent Linear Quantile Regressor."""

import os

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from conformal_tights._coherent_linear_quantile_regressor import CoherentLinearQuantileRegressor


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip on GitHub Actions")
def test_sklearn_check_estimator() -> None:
    """Check that the meta-estimator conforms to sklearn's standards."""
    model = CoherentLinearQuantileRegressor(quantiles=np.array([0.5]))
    check_estimator(model)
