"""Test the Conformal Coherent Quantile Regressor."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from xgboost import XGBRegressor

from conformal_tights import ConformalCoherentQuantileRegressor
from tests.conftest import Dataset


def test_conformal_quantile_regressor_coverage(dataset: Dataset, regressor: BaseEstimator) -> None:
    """Test ConformalCoherentQuantileRegressor's coverage."""
    # Unpack the dataset.
    X_train, X_test, y_train, y_test = dataset
    # Train the models.
    model = ConformalCoherentQuantileRegressor(estimator=regressor)
    model.fit(X_train, y_train)
    # Verify the coherence of the predicted quantiles.
    ŷ_quantiles = model.predict(X_test, quantiles=np.linspace(0.1, 0.9, 3))
    for j in range(ŷ_quantiles.shape[1] - 1):
        assert np.all(ŷ_quantiles.iloc[:, j] <= ŷ_quantiles.iloc[:, j + 1])
    # Verify the coverage of the predicted intervals.
    for desired_coverage in (0.7, 0.8, 0.9):
        ŷ_interval = model.predict(X_test, coverage=desired_coverage)
        assert np.all(ŷ_interval.iloc[:, 0] <= ŷ_interval.iloc[:, 1])
        covered = (ŷ_interval.iloc[:, 0] <= y_test) & (y_test <= ŷ_interval.iloc[:, 1])
        actual_coverage = np.mean(covered)
        assert actual_coverage >= 0.97 * desired_coverage


def test_sklearn_check_estimator() -> None:
    """Check that the meta-estimator conforms to sklearn's standards."""
    model = ConformalCoherentQuantileRegressor(estimator=XGBRegressor())
    check_estimator(model)
