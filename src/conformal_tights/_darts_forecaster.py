"""Darts Forecaster."""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.regression_model import (
    FUTURE_LAGS_TYPE,
    LAGS_TYPE,
    RegressionModel,
    RegressionModelWithCategoricalCovariates,
    _LikelihoodMixin,
)
from sklearn.utils import check_random_state

from conformal_tights._conformal_coherent_quantile_regressor import (
    ConformalCoherentQuantileRegressor,
)
from conformal_tights._typing import FloatMatrix, FloatTensor, FloatVector

F = TypeVar("F", np.float32, np.float64)


class _DartsAdapter:
    def __init__(
        self, model: ConformalCoherentQuantileRegressor, quantile: float, quantiles: npt.ArrayLike
    ):
        self.model = model
        self.quantile = quantile
        self.quantiles = np.asarray(quantiles)

    def predict(self, x: pd.DataFrame, **kwargs: Any) -> FloatMatrix[F]:
        # Call ConformalCoherentQuantileRegressor's predict_quantiles.
        q = np.asarray(self.model.predict_quantiles(x, quantiles=self.quantiles))
        # Filter out the requested quantile.
        q = q[:, self.quantiles == self.quantile]
        return q


class DartsForecaster(_LikelihoodMixin, RegressionModel):
    def __init__(  # noqa: PLR0913
        self,
        # Regressor used by Darts to produce probabilistic forecasts.
        model: ConformalCoherentQuantileRegressor,
        *,
        # Default darts.models.RegressionModel parameters.
        lags: LAGS_TYPE | None = None,
        lags_past_covariates: LAGS_TYPE | None = None,
        lags_future_covariates: FUTURE_LAGS_TYPE | None = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: dict[str, Any] | None = None,
        multi_models: bool | None = True,
        use_static_covariates: bool = True,
        # Default darts.models.RegressionModelWithCategoricalCovariates parameters.
        categorical_past_covariates: str | list[str] | None = None,
        categorical_future_covariates: str | list[str] | None = None,
        categorical_static_covariates: str | list[str] | None = None,
    ) -> None:
        """Initialize a Darts Conformal Coherent Quantile Regressor."""
        # Initialise _LikelihoodMixin.
        self._likelihood = "quantile"
        self._model_container = self._get_model_container()
        self._rng = check_random_state(model.random_state)  # Generator for sampling.
        # Initialise darts.models.RegressionModel.
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            multi_models=multi_models,
            model=model,
            use_static_covariates=use_static_covariates,
        )
        # Initialise darts.models.RegressionModelWithCategoricalCovariates.
        self.categorical_past_covariates = (
            [categorical_past_covariates]
            if isinstance(categorical_past_covariates, str)
            else categorical_past_covariates
        )
        self.categorical_future_covariates = (
            [categorical_future_covariates]
            if isinstance(categorical_future_covariates, str)
            else categorical_future_covariates
        )
        self.categorical_static_covariates = (
            [categorical_static_covariates]
            if isinstance(categorical_static_covariates, str)
            else categorical_static_covariates
        )

    @property
    def likelihood(self) -> str | None:
        return getattr(self, "_likelihood", None)

    def _create_lagged_data(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries],
        future_covariates: Sequence[TimeSeries],
        max_samples_per_ts: int,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, FloatVector[F]] | tuple[pd.DataFrame, FloatVector[F], FloatVector[F]]:
        """Override training data to add support for categorical covariates."""
        # Validate categoricals with RegressionModelWithCategoricalCovariates. We cannot inherit
        # from RegressionModelWithCategoricalCovariates because it was developed with LightGBM in
        # mind and does not support other regressors like XGBRegressor.
        RegressionModelWithCategoricalCovariates._validate_categorical_covariates(  # noqa: SLF001
            self,
            series,
            past_covariates,
            future_covariates,
        )
        # Identify which columns in the lagged data are categorical.
        cat_col_indices, _ = RegressionModelWithCategoricalCovariates._get_categorical_features(  # noqa: SLF001
            self,
            series,
            past_covariates,
            future_covariates,
        )
        # Create lagged training data.
        training_samples, *training_labels_and_sample_weights = super()._create_lagged_data(
            series, past_covariates, future_covariates, max_samples_per_ts, **kwargs
        )
        # Convert categorical columns to pd.Categorical so that the wrapped regressor can handle
        # them appropriately.
        self.cat_col_categories_: dict[float, pd.Index] = {}
        training_samples_df = pd.DataFrame(training_samples)
        cols = training_samples_df.columns
        for cat_col_index in cat_col_indices:
            cat_col = training_samples_df[cols[cat_col_index]].astype("category")
            self.cat_col_categories_[cat_col_index] = cat_col.cat.categories
            training_samples_df[cols[cat_col_index]] = cat_col
        # Store the (modified) model for filling the model container in _predict_and_sample.
        self.central_model_ = self.model
        return training_samples_df, *training_labels_and_sample_weights

    def _predict_and_sample(
        self,
        x: FloatMatrix[F],
        num_samples: int,
        predict_likelihood_parameters: bool,  # noqa: FBT001
        quantiles: npt.ArrayLike = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
        **kwargs: Any,
    ) -> FloatMatrix[F] | FloatTensor[F]:
        """Override inference data to add support for categorical covariates."""
        # Instead of choosing the quantiles at initialisation time, allow users to set the quantiles
        # of DartsForecaster.predict at prediction time.
        if getattr(self, "quantiles", None) != quantiles:
            self.quantiles, self._median_idx = self._prepare_quantiles(quantiles)
            self._model_container.clear()
            for quantile in self.quantiles:
                self._model_container[quantile] = _DartsAdapter(
                    self.central_model_, quantile, self.quantiles
                )
        # Convert categorical columns to pd.Categorical so that the wrapped regressor can handle
        # them appropriately.
        x_df = pd.DataFrame(x)
        for cat_col_index, cat_col_categories in self.cat_col_categories_.items():
            x_df[x_df.columns[cat_col_index]] = pd.Categorical(
                x_df[x_df.columns[cat_col_index]], categories=cat_col_categories
            )
        # Call _LikelihoodMixin._predict_and_sample_likelihood to enable probabilistic forecasting.
        outputs: FloatMatrix[F] | FloatTensor[F] = self._predict_and_sample_likelihood(
            x_df, num_samples, self.likelihood, predict_likelihood_parameters, **kwargs
        )
        return outputs

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """Indicate that this is a probabilistic model for darts > 0.28."""
        return True

    @property
    def _is_probabilistic(self) -> bool:
        """Indicate that this is a probabilistic model for darts <= 0.28."""
        return True
