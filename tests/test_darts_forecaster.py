"""Test the Darts Forecaster."""

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml

from conformal_tights import ConformalCoherentQuantileRegressor, DartsForecaster
from conformal_tights._darts_forecaster import _DartsAdapter


def test_darts_forecaster_coverage(regressor: BaseEstimator) -> None:
    """Test DartsForecaster's coverage."""
    # Fetch the dataset.
    X, y = fetch_openml(
        name="Bike_Sharing_Demand", version=7, return_X_y=True, as_frame=True, parser="auto"
    )
    # Convert the dataset to a Darts.Timeseries.
    target_series = TimeSeries.from_series(y)
    covariates_series = TimeSeries.from_dataframe(X)
    # Split in train and test.
    target_train, target_test = train_test_split(target_series, test_size=0.15)
    feature_train, feature_test = train_test_split(covariates_series, test_size=0.15)
    # Create a DartsForecaster.
    model = ConformalCoherentQuantileRegressor(estimator=regressor)
    forecaster = DartsForecaster(
        model=model,
        lags=48,
        lags_future_covariates=[0],
        categorical_future_covariates=X.select_dtypes(include=["category"]).columns.tolist(),
    )
    forecaster.fit(target_train, future_covariates=feature_train)
    # Make a probabilistic forecast.
    quantiles = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
    forecast_horizon = 48
    forecast = forecaster.predict(
        n=forecast_horizon, future_covariates=feature_test, num_samples=500, quantiles=quantiles
    )
    # Verify the coherence of the predicted quantiles.
    ŷ_quantiles = forecast.quantile(quantiles).to_dataframe()
    for j in range(ŷ_quantiles.shape[1] - 1):
        assert np.all(ŷ_quantiles.iloc[:, j] <= ŷ_quantiles.iloc[:, j + 1])
    # Verify the coverage of the predicted intervals.
    y_test = target_test.to_series().iloc[: ŷ_quantiles.shape[0]]
    for j in range((len(quantiles) - 1) // 2):
        desired_coverage = quantiles[-(j + 1)] - quantiles[j]
        ŷ_interval = ŷ_quantiles.iloc[:, [j, -(j + 1)]]
        assert np.all(ŷ_interval.iloc[:, 0] <= ŷ_interval.iloc[:, 1])
        covered = (ŷ_interval.iloc[:, 0] <= y_test) & (y_test <= ŷ_interval.iloc[:, 1])
        actual_coverage = np.mean(covered)
        assert actual_coverage >= 0.97 * desired_coverage


def test_darts_forecaster_reuses_quantile_predictions() -> None:
    """Test that all quantile adapters share one predict_quantiles() result per prediction call."""
    model = ConformalCoherentQuantileRegressor()
    calls = 0

    def predict_quantiles(x: pd.DataFrame, quantiles: list[float]) -> np.ndarray:
        nonlocal calls
        calls += 1
        np.testing.assert_array_equal(quantiles, [0.1, 0.5, 0.9], strict=True)
        assert isinstance(x, pd.DataFrame)
        return np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    model.predict_quantiles = predict_quantiles  # type: ignore[assignment]
    adapters = _DartsAdapter.model_container(model, [0.1, 0.5, 0.9])
    x = pd.DataFrame({"feature": [0.0, 1.0]})
    np.testing.assert_array_equal(adapters[0.1].predict(x), [[1.0], [10.0]])
    np.testing.assert_array_equal(adapters[0.5].predict(x), [[2.0], [20.0]])
    np.testing.assert_array_equal(adapters[0.9].predict(x), [[3.0], [30.0]])
    assert calls == 1
