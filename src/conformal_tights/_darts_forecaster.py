"""Darts Forecaster."""

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from darts.models.forecasting.sklearn_model import (
    FUTURE_LAGS_TYPE,
    LAGS_TYPE,
    SKLearnModelWithCategoricalFeatures,
)
from darts.utils.likelihood_models.sklearn import QuantileRegression

from conformal_tights._conformal_coherent_quantile_regressor import (
    ConformalCoherentQuantileRegressor,
)
from conformal_tights._typing import FloatMatrix, FloatTensor

F = TypeVar("F", np.float32, np.float64)


class _DartsAdapter:
    @dataclass
    class PredictCache:
        x: pd.DataFrame | None = None
        q: FloatMatrix[np.float64] | None = None

    def __init__(
        self,
        model: ConformalCoherentQuantileRegressor,
        quantile: float,
        quantiles: npt.ArrayLike,
        last_prediction: "_DartsAdapter.PredictCache",
    ):
        self.model = model
        self.quantile = quantile
        self.quantiles = np.asarray(quantiles)
        self.last_prediction = last_prediction

    @classmethod
    def model_container(
        cls, model: ConformalCoherentQuantileRegressor, quantiles: list[float]
    ) -> dict[float, "_DartsAdapter"]:
        last_prediction = cls.PredictCache()
        return {
            quantile: cls(model, quantile, quantiles, last_prediction) for quantile in quantiles
        }

    def predict(self, x: pd.DataFrame, **kwargs: Any) -> FloatMatrix[np.float64]:
        # Reuse the full quantile prediction for all adapters in the model container.
        q = self.last_prediction.q
        if self.last_prediction.x is not x or q is None:
            q = np.asarray(self.model.predict_quantiles(x, quantiles=self.quantiles))
            self.last_prediction.x = x
            self.last_prediction.q = q
        # Filter out the requested quantile.
        return q[:, self.quantiles == self.quantile]


class DartsForecaster(SKLearnModelWithCategoricalFeatures):
    def __init__(  # noqa: PLR0913
        self,
        # Regressor used by Darts to produce probabilistic forecasts.
        model: ConformalCoherentQuantileRegressor,
        *,
        # Default SKLearnModelWithCategoricalFeatures parameters.
        lags: LAGS_TYPE | None = None,
        lags_past_covariates: LAGS_TYPE | None = None,
        lags_future_covariates: FUTURE_LAGS_TYPE | None = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: dict[str, Any] | None = None,
        multi_models: bool | None = True,
        use_static_covariates: bool = True,
        categorical_past_covariates: str | list[str] | None = None,
        categorical_future_covariates: str | list[str] | None = None,
        categorical_static_covariates: str | list[str] | None = None,
        random_state: int | None = 42,
    ) -> None:
        """Initialize a Darts Conformal Coherent Quantile Regressor."""
        # Initialise darts.models.SKLearnModelWithCategoricalFeatures.
        super().__init__(
            model=model,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            categorical_past_covariates=categorical_past_covariates,
            categorical_future_covariates=categorical_future_covariates,
            categorical_static_covariates=categorical_static_covariates,
            random_state=random_state,
        )
        # Initialise probabilistic forecasting for darts >= 0.39.
        self._likelihood = QuantileRegression(n_outputs=self.pred_dim, quantiles=[0.5])
        self._model_container: dict[float, _DartsAdapter] = {}

    @property
    def _categorical_fit_param(self) -> str | None:
        """No categorical fit parameter is required for the wrapped regressor."""
        return None

    def _format_samples(
        self, samples: npt.NDArray[Any], labels: npt.NDArray[Any] | None = None
    ) -> tuple[pd.DataFrame | npt.NDArray[Any], npt.NDArray[Any] | None]:
        """Convert categorical columns to pd.Categorical for the wrapped regressor."""
        samples, labels = super()._format_samples(samples, labels)
        if len(self._categorical_indices) == 0:
            return samples, labels
        samples_df = pd.DataFrame(samples)
        if labels is not None and not hasattr(self, "cat_col_categories_"):
            self.cat_col_categories_: dict[int, pd.Index] = {}
        for cat_col_index in self._categorical_indices:
            col = samples_df.columns[cat_col_index]
            values = samples_df[col].astype("string")  # For XGBoost compatibility.
            if labels is None:  # Predict: reuse training categories.
                samples_df[col] = pd.Categorical(
                    values, categories=self.cat_col_categories_[cat_col_index]
                )
            else:  # Fit: store categories.
                cat_col = values.astype("category")
                self.cat_col_categories_[cat_col_index] = cat_col.cat.categories
                samples_df[col] = cat_col
        return samples_df, labels

    def _predict(
        self,
        x: FloatMatrix[F],
        num_samples: int,
        predict_likelihood_parameters: bool,  # noqa: FBT001
        quantiles: npt.ArrayLike = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
        **kwargs: Any,
    ) -> FloatTensor[F]:
        """Generate probabilistic predictions with `self._likelihood`."""
        # Instead of choosing the quantiles at initialisation time, allow users to set the quantiles
        # of DartsForecaster.predict at prediction time.
        if not np.array_equal(np.asarray(getattr(self, "quantiles", None)), np.asarray(quantiles)):
            quantiles_array = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
            self.quantiles = sorted(float(quantile) for quantile in quantiles_array.tolist())
            self._median_idx = self.quantiles.index(0.5)
            self._likelihood = QuantileRegression(n_outputs=self.pred_dim, quantiles=self.quantiles)
            self._model_container = _DartsAdapter.model_container(self.model, self.quantiles)
        outputs: FloatTensor[F] = super()._predict(
            x, num_samples, predict_likelihood_parameters, **kwargs
        )
        return outputs
