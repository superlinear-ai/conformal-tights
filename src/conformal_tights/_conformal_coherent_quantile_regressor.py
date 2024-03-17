"""Conformal Coherent Quantile Regressor meta-estimator."""

from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_X_y,
)
from xgboost import XGBRegressor

from conformal_tights._coherent_linear_quantile_regressor import CoherentLinearQuantileRegressor
from conformal_tights._typing import FloatMatrix, FloatVector

if TYPE_CHECKING:
    import pandas as pd

F = TypeVar("F", np.float32, np.float64)


class ConformalCoherentQuantileRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """Conformal Coherent Quantile Regressor meta-estimator.

    Adds conformally calibrated quantile and interval prediction to a given regressor by fitting a
    meta-estimator as follows:

        1. The given data is split into a training set and a conformal calibration set.
        2. The training set is used to fit the given regressor.
        3. The training set is also used to fit a nonconformity estimator, which is by default an
           XGBoost vector quantile regressor for the quantiles (1/8, 1/4, 1/2, 3/4, 7/8). These
           quantiles are not necessarily monotonic and may cross each other.
        4. The conformal calibration set is split into two levels.
        5. The level 1 conformal calibration set is used to fit a Coherent Linear Quantile
           Regression model of the (relative) residuals given the level 1 nonconformity estimates.
           This model produces conformally calibrated quantiles of the (relative) residuals that are
           coherent in the sense that they increase monotonically.
        6. The level 2 conformal calibration set is used to fit a per-quantile conformal bias on top
           of the level 1 conformal quantile predictions of the (relative) residuals.

    Quantile and interval predictions are made by predicting the nonconformity estimates, converting
    those into conformally calibrated and coherent quantiles, and then adding a conformally
    calibrated bias to the result. At the user's request, the bias can prioritize quantile accuracy
    or interval coverage.

    The level 1 and level 2 conformal predictors are lazily fitted on both the absolute and relative
    residuals for the requested quantiles at prediction time. This allows the user to choose which
    quantiles to predict, and to select the quantile predictions with the lowest dispersion.
    """

    def __init__(  # noqa: PLR0913
        self,
        estimator: BaseEstimator | Literal["auto"] = "auto",
        *,
        nonconformity_estimator: BaseEstimator | Literal["auto"] = "auto",
        nonconformity_quantiles: npt.ArrayLike = (1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8),
        conformal_calibration_size: tuple[float, int] = (0.3, 1440),
        random_state: int | np.random.RandomState | None = 42,
    ) -> None:
        """Initialize the Conformal Coherent Quantile Regressor.

        Parameters
        ----------
        estimator
            The regressor to wrap, used for point prediction. If "auto", uses an `XGBRegressor`.
        nonconformity_estimator
            A nonconformity estimator to use. If "auto", uses XGBoost's vector quantile regressor
            for the given `nonconformity_quantiles`.
        nonconformity_quantiles
            The quantiles that the nonconformity estimator should predict when
            `nonconformity_estimator` is "auto".
        conformal_calibration_size
            A tuple of the relative and absolute size of the conformal calibration set. The smallest
            of the two is used.
        random_state
            The random state to use for reproducibility.
        """
        self.estimator = estimator
        self.nonconformity_estimator = nonconformity_estimator
        self.nonconformity_quantiles = nonconformity_quantiles
        self.conformal_calibration_size = conformal_calibration_size
        self.random_state = random_state

    def fit(
        self,
        X: "FloatMatrix[F] | pd.DataFrame",
        y: "FloatVector[F] | pd.Series",
        *,
        sample_weight: "FloatVector[F] | pd.Series | None" = None,
    ) -> "ConformalCoherentQuantileRegressor":
        """Fit this predictor."""
        # Validate input.
        check_X_y(X, y, force_all_finite=False, ensure_min_samples=3, y_numeric=True)
        # Learn dimensionality and dtypes.
        if not hasattr(X, "dtypes"):
            X = np.asarray(X)
        y = np.ravel(np.asarray(y))
        self.n_features_in_: int = X.shape[1]
        self.y_dtype_: npt.DTypeLike = y.dtype  # Used to cast predictions to the correct dtype.
        if np.all(y.astype(np.intp) == y):
            self.y_dtype_ = np.intp  # To satisfy sklearn's `check_regressors_int`.
        y = y.astype(np.float64)  # To support datetime64[ns] and timedelta64[ns].
        if sample_weight is not None:
            check_consistent_length(y, sample_weight)
            sample_weight = np.ravel(np.asarray(sample_weight).astype(np.float64))
        # Use the smallest of the relative and absolute calibration sizes.
        calib_size = min(
            int(self.conformal_calibration_size[0] * X.shape[0]), self.conformal_calibration_size[1]
        )
        # Split input into training and conformal calibration sets.
        X_train, self.X_calib_, y_train, self.y_calib_, *sample_weights = train_test_split(
            X,
            y,
            *([sample_weight] if sample_weight is not None else []),
            test_size=calib_size,
            random_state=self.random_state,
        )
        sample_weight_train, sample_weight_calib = (
            sample_weights[:2] if sample_weight is not None else (None, None)
        )
        # Split the conformal calibration set into two levels.
        X_calib_l1, X_calib_l2, y_calib_l1, y_calib_l2, *sample_weights_calib = train_test_split(
            self.X_calib_,
            self.y_calib_,
            *([sample_weight_calib] if sample_weight_calib is not None else []),
            test_size=self.conformal_calibration_size[0],
            random_state=self.random_state,
        )
        self.sample_weight_calib_l1_, self.sample_weight_calib_l2_ = (
            sample_weights_calib[:2] if sample_weight is not None else (None, None)  # type: ignore[has-type]
        )
        # Fit the given estimator on the training data.
        self.estimator_ = (
            clone(self.estimator)
            if self.estimator != "auto"
            else XGBRegressor(objective="reg:absoluteerror")
        )
        if isinstance(self.estimator_, XGBRegressor):
            self.estimator_.set_params(enable_categorical=True, random_state=self.random_state)
        self.estimator_.fit(X_train, y_train, sample_weight=sample_weight_train)
        # Fit a nonconformity estimator on the training data with XGBRegressor's vector quantile
        # regression. We fit a minimal number of quantiles to reduce the computational cost, but
        # also to reduce the risk of overfitting in the coherent quantile regressor that is applied
        # on top of the nonconformity estimates.
        self.nonconformity_estimator_ = (
            clone(self.nonconformity_estimator)
            if self.nonconformity_estimator != "auto"
            else XGBRegressor()
        )
        if isinstance(self.nonconformity_estimator_, XGBRegressor):
            self.nonconformity_estimator_.set_params(
                objective="reg:quantileerror",
                quantile_alpha=self.nonconformity_quantiles,
                enable_categorical=True,
                random_state=self.random_state,
            )
        self.nonconformity_estimator_.fit(X_train, y_train, sample_weight=sample_weight_train)
        # Predict on the level 1 calibration set.
        self.ŷ_calib_l1_ = self.estimator_.predict(X_calib_l1)
        self.ŷ_calib_l1_nonconformity_ = self.nonconformity_estimator_.predict(X_calib_l1)
        self.residuals_calib_l1_ = self.ŷ_calib_l1_ - y_calib_l1
        # Predict on the level 2 calibration set.
        self.ŷ_calib_l2_ = self.estimator_.predict(X_calib_l2)
        self.ŷ_calib_l2_nonconformity_ = self.nonconformity_estimator_.predict(X_calib_l2)
        self.residuals_calib_l2_ = self.ŷ_calib_l2_ - y_calib_l2
        # Lazily fit level 1 conformal predictors as coherent linear quantile regression models that
        # predict quantiles of the (relative) residuals given the nonconformity estimates, and
        # level 2 conformal biases.
        self.conformal_l1_: dict[str, dict[tuple[float, ...], CoherentLinearQuantileRegressor]] = {
            "Δŷ": {},
            "Δŷ/ŷ": {},
        }
        self.conformal_l2_: dict[str, dict[tuple[float, ...], FloatVector[F]]] = {
            "Δŷ": {},
            "Δŷ/ŷ": {},
        }
        return self

    def _lazily_fit_conformal_predictor(
        self, target_type: str, quantiles: npt.ArrayLike
    ) -> tuple[CoherentLinearQuantileRegressor, FloatVector[F]]:
        """Lazily fit a conformal predictor for a given array of quantiles."""
        quantiles = np.asarray(quantiles)
        quantiles_tuple = tuple(quantiles)
        if quantiles_tuple in self.conformal_l1_[target_type]:
            # Retrieve level 1 and level 2.
            cqr_l1 = self.conformal_l1_[target_type][quantiles_tuple]
            bias_l2 = self.conformal_l2_[target_type][quantiles_tuple]
        else:
            # Fit level 1: a coherent quantile regressor that predicts quantiles of the (relative)
            # residuals.
            eps = np.finfo(self.ŷ_calib_l1_.dtype).eps
            abs_ŷ_calib_l1 = np.maximum(np.abs(self.ŷ_calib_l1_), eps)
            X_cqr = self.ŷ_calib_l1_nonconformity_
            y_cqr = self.residuals_calib_l1_ / (abs_ŷ_calib_l1 if "/ŷ" in target_type else 1)
            cqr_l1 = CoherentLinearQuantileRegressor(quantiles=quantiles)
            cqr_l1.fit(X_cqr, y_cqr, sample_weight=self.sample_weight_calib_l1_)
            self.conformal_l1_[target_type][quantiles_tuple] = cqr_l1
            # Fit level 2: a per-quantile conformal bias on top of the level 1 conformal quantile
            # predictions of the (relative) residuals.
            abs_ŷ_calib_l2 = np.maximum(np.abs(self.ŷ_calib_l2_), eps)
            Δŷ_calib_l2_quantiles = cqr_l1.predict(self.ŷ_calib_l2_nonconformity_)
            bias_l2 = np.empty(quantiles.shape, dtype=self.ŷ_calib_l1_.dtype)
            for j, quantile in enumerate(quantiles):
                bias_l2[j] = np.quantile(
                    -(
                        (self.residuals_calib_l2_ / (abs_ŷ_calib_l2 if "/ŷ" in target_type else 1))
                        + Δŷ_calib_l2_quantiles[:, j]
                    ),
                    quantile,
                )
            self.conformal_l2_[target_type][quantiles_tuple] = bias_l2
        return cqr_l1, bias_l2  # type: ignore[return-value]

    @overload
    def predict_quantiles(
        self,
        X: FloatMatrix[F],
        *,
        quantiles: npt.ArrayLike,
        priority: Literal["accuracy", "coverage"] = "accuracy",
    ) -> FloatMatrix[F]: ...

    @overload
    def predict_quantiles(
        self,
        X: "pd.DataFrame",
        *,
        quantiles: npt.ArrayLike,
        priority: Literal["accuracy", "coverage"] = "accuracy",
    ) -> "pd.DataFrame": ...

    def predict_quantiles(
        self,
        X: "FloatMatrix[F] | pd.DataFrame",
        *,
        quantiles: npt.ArrayLike,
        priority: Literal["accuracy", "coverage"] = "accuracy",
    ) -> "FloatMatrix[F] | pd.DataFrame":
        """Predict conformally calibrated quantiles on a given dataset."""
        # Predict the absolute and relative quantiles.
        quantiles = np.asarray(quantiles)
        ŷ = np.asarray(self.estimator_.predict(X))
        X_cqr = self.nonconformity_estimator_.predict(X)
        cqr_abs, bias_abs = self._lazily_fit_conformal_predictor("Δŷ", quantiles)
        cqr_rel, bias_rel = self._lazily_fit_conformal_predictor("Δŷ/ŷ", quantiles)
        if priority == "coverage":  # Only allow quantile expansion when the priority is coverage.
            center = 0.5
            bias_abs[center <= quantiles] = np.maximum(bias_abs[center <= quantiles], 0)
            bias_abs[quantiles <= center] = np.minimum(bias_abs[quantiles <= center], 0)
            bias_rel[center <= quantiles] = np.maximum(bias_rel[center <= quantiles], 0)
            bias_rel[quantiles <= center] = np.minimum(bias_rel[quantiles <= center], 0)
        Δŷ_quantiles = np.dstack(
            [
                cqr_abs.predict(X_cqr) + bias_abs[np.newaxis, :],
                np.abs(ŷ[:, np.newaxis]) * (cqr_rel.predict(X_cqr) + bias_rel[np.newaxis, :]),
            ]
        )
        # Choose between the the absolute and relative quantiles for each example in order to
        # minimise the dispersion of the predicted quantiles.
        dispersion = np.std(Δŷ_quantiles, axis=1)
        Δŷ_quantiles = Δŷ_quantiles[
            np.arange(Δŷ_quantiles.shape[0]), :, np.argmin(dispersion, axis=-1)
        ]
        ŷ_quantiles: FloatMatrix[F] = (ŷ[:, np.newaxis] + Δŷ_quantiles).astype(self.y_dtype_)
        # Convert ŷ_quantiles to a pandas DataFrame if X is a pandas DataFrame.
        if hasattr(X, "dtypes") and hasattr(X, "index"):
            try:
                import pandas as pd
            except ImportError:
                pass
            else:
                ŷ_quantiles_df = pd.DataFrame(ŷ_quantiles, index=X.index, columns=quantiles)
                ŷ_quantiles_df.columns.name = "quantile"
                return ŷ_quantiles_df
        return ŷ_quantiles

    @overload
    def predict_interval(self, X: FloatMatrix[F], *, coverage: float = 0.95) -> FloatMatrix[F]: ...

    @overload
    def predict_interval(self, X: "pd.DataFrame", *, coverage: float = 0.95) -> "pd.DataFrame": ...

    def predict_interval(
        self, X: "FloatMatrix[F] | pd.DataFrame", *, coverage: float = 0.95
    ) -> "FloatMatrix[F] | pd.DataFrame":
        """Predict conformally calibrated intervals on a given dataset."""
        # Convert the coverage probability to lower and upper quantiles.
        lb = (1 - coverage) / 2
        ub = 1 - lb
        # Compute the prediction interval with predict_quantiles.
        ŷ_quantiles = self.predict_quantiles(X, quantiles=(lb, ub), priority="coverage")
        return ŷ_quantiles

    @overload
    def predict(
        self, X: FloatMatrix[F], *, coverage: None = None, quantiles: None = None
    ) -> FloatVector[F]: ...

    @overload
    def predict(
        self, X: FloatMatrix[F], *, coverage: float, quantiles: None = None
    ) -> FloatMatrix[F]: ...

    @overload
    def predict(
        self, X: FloatMatrix[F], *, coverage: None = None, quantiles: npt.ArrayLike
    ) -> FloatMatrix[F]: ...

    @overload
    def predict(
        self, X: "pd.DataFrame", *, coverage: None = None, quantiles: None = None
    ) -> "pd.Series": ...

    @overload
    def predict(
        self, X: "pd.DataFrame", *, coverage: float, quantiles: None = None
    ) -> "pd.DataFrame": ...

    @overload
    def predict(
        self, X: "pd.DataFrame", *, coverage: None = None, quantiles: npt.ArrayLike
    ) -> "pd.DataFrame": ...

    def predict(
        self,
        X: "FloatMatrix[F] | pd.DataFrame",
        *,
        coverage: float | None = None,
        quantiles: npt.ArrayLike | None = None,
    ) -> "FloatVector[F] | pd.Series | FloatMatrix[F] | pd.DataFrame":
        """Predict on a given dataset."""
        assert coverage is None or quantiles is None
        check_is_fitted(self)
        check_array(X, force_all_finite=False)
        if coverage is not None:
            ŷ_interval = self.predict_interval(X, coverage=coverage)
            return ŷ_interval
        if quantiles is not None:
            ŷ_quantiles = self.predict_quantiles(X, quantiles=quantiles)
            return ŷ_quantiles
        ŷ = self.estimator_.predict(X).astype(self.y_dtype_)
        if hasattr(X, "dtypes") and hasattr(X, "index"):
            try:
                import pandas as pd
            except ImportError:
                pass
            else:
                ŷ_series = pd.Series(ŷ, index=X.index)
                return ŷ_series
        return ŷ

    def _more_tags(self) -> dict[str, bool]:
        """Return more tags for the estimator."""
        return {"allow_nan": True}
