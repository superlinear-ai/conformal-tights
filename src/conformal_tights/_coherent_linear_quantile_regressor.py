"""Coherent Linear Quantile Regressor."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

# Try to import validate_data (available in sklearn >= 1.6)
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    # For sklearn < 1.6
    from sklearn.utils.validation import check_array, check_X_y

    validate_data = None

from conformal_tights._typing import FloatMatrix, FloatVector

F = TypeVar("F", np.float32, np.float64)


def coherent_linear_quantile_regression(
    X: FloatMatrix[F],
    y: FloatVector[F],
    *,
    quantiles: FloatVector[F],
    sample_weight: FloatVector[F] | None = None,
    coherence_buffer: int = 3,
) -> tuple[FloatMatrix[F], FloatMatrix[F]]:
    """Solve a Coherent Linear Quantile Regression problem.

    Minimizes the quantile loss:

        ∑ᵢ,ⱼ {
                 qⱼ (yᵢ - ŷ⁽ʲ⁾ᵢ) : yᵢ ≥ ŷ⁽ʲ⁾ᵢ,
            (1 - qⱼ)(ŷ⁽ʲ⁾ᵢ - yᵢ) : ŷ⁽ʲ⁾ᵢ > yᵢ
        }

    for the linear model ŷ⁽ʲ⁾ := Xβ⁽ʲ⁾, given an input dataset X, target y, and quantile ranks qⱼ.

    We achieve so-called 'coherent' quantiles by enforcing monotonicity of the predicted quantiles
    with the constraint Xβ⁽ʲ⁾ ≤ Xβ⁽ʲ⁺¹⁾ for each consecutive pair of quantile ranks in an extended
    set of quantile ranks that comprises the requested quantile ranks and a number of auxiliary
    quantile ranks in between.

    The optimization problem is formulated as a linear program by introducing the auxiliary residual
    vectors Δ⁽ʲ⁾⁺, Δ⁽ʲ⁾⁻ ≥ 0 so that Xβ⁽ʲ⁾ - y = Δ⁽ʲ⁾⁺ - Δ⁽ʲ⁾⁻. The objective then becomes
    ∑ᵢ,ⱼ qⱼΔ⁽ʲ⁾⁻ᵢ + (1 - qⱼ)Δ⁽ʲ⁾⁺ᵢ + αt⁽ʲ⁾ᵢ for t⁽ʲ⁾ := |β⁽ʲ⁾|. The L1 regularization parameter α is
    automatically determined to minimize the impact on the solution β.

    Parameters
    ----------
    X
        The feature matrix.
    y
        The target values.
    quantiles
        The quantiles to estimate (between 0 and 1).
    sample_weight
        The optional sample weight to use for each sample.
    coherence_buffer
        The number of auxiliary quantiles to introduce. Smaller is faster, larger yields more
        coherent quantiles.

    Returns
    -------
    β
        The estimated regression coefficients so that Xβ produces quantile predictions ŷ.
    β_full
        The estimated regression coefficients including all auxiliary quantiles.
    """
    # Learn the input dimensions.
    num_samples, num_features = X.shape
    # Add buffer quantile ranks in between the given quantile ranks so that we have an even stronger
    # guarantee on the monotonicity of the predicted quantiles.
    quantiles = np.interp(
        x=np.linspace(0, len(quantiles) - 1, (len(quantiles) - 1) * (1 + coherence_buffer) + 1),
        xp=np.arange(len(quantiles)),
        fp=quantiles,
    ).astype(quantiles.dtype)
    num_quantiles = len(quantiles)
    # Validate the input.
    assert np.array_equal(quantiles, np.sort(quantiles)), "Quantile ranks must be sorted."
    assert sample_weight is None or np.all(sample_weight >= 0), "Sample weights must be >= 0."
    # Handle sample weights by explicitly repeating samples
    # This approach directly mimics how sklearn's check_sample_weight_equivalence works
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        # Filter out samples with zero weight (equivalent to removing them)
        mask = sample_weight > 0
        if not np.all(mask):
            X = X[mask]
            y = y[mask]
            sample_weight = sample_weight[mask]

        # For each sample with weight > 1, repeat it weight-1 times
        # (it's already in the dataset once)
        repeat_mask = sample_weight > 1
        if np.any(repeat_mask):
            X_repeat: list[FloatMatrix[F]] = []
            y_repeat: list[FloatVector[F]] = []

            for i in np.where(repeat_mask)[0]:
                # Repeat each sample (weight-1) times (it's already in the dataset once)
                repeat_count = int(sample_weight[i]) - 1
                X_repeat.append(np.tile(X[i : i + 1], (repeat_count, 1)))
                y_repeat.append(np.tile(y[i], repeat_count))

            if X_repeat:
                X = np.vstack([X, *X_repeat])
                y = np.hstack([y, *y_repeat])

        # Now all samples have effectively weight=1
        sample_weight = None

    # Update dimensions after potential sample repetition
    num_samples = X.shape[0]

    eps = np.finfo(y.dtype).eps
    α = np.sqrt(eps) / (num_quantiles * num_features)

    # With sample repetition handled above, use uniform weights for all samples
    sample_weight = np.ones(num_samples, dtype=y.dtype)

    # Construct the objective function coefficients
    delta_plus_weights = np.zeros(num_quantiles * num_samples, dtype=y.dtype)
    delta_minus_weights = np.zeros(num_quantiles * num_samples, dtype=y.dtype)

    for j in range(num_quantiles):
        start_idx = j * num_samples
        end_idx = (j + 1) * num_samples
        delta_plus_weights[start_idx:end_idx] = (1 - quantiles[j]) / num_quantiles
        delta_minus_weights[start_idx:end_idx] = quantiles[j] / num_quantiles

    # Complete objective function
    c = np.hstack(
        [
            np.zeros(num_quantiles * num_features, dtype=y.dtype),  # β⁽ʲ⁾ for each qⱼ
            α * np.ones(num_quantiles * num_features, dtype=y.dtype),  # t⁽ʲ⁾ for each qⱼ
            delta_plus_weights,  # Δ⁽ʲ⁾⁺ with uniform weights (since we repeated samples)
            delta_minus_weights,  # Δ⁽ʲ⁾⁻ with uniform weights (since we repeated samples)
        ]
    )
    # Construct the equation constraints Xβ⁽ʲ⁾ - y = Δ⁽ʲ⁾⁺ - Δ⁽ʲ⁾⁻ for each quantile rank qⱼ.
    # For samples with zero weights, we'll still include them in the constraints but their
    # contribution to the objective function will be zero, so they won't affect the solution.
    A_eq = sparse.hstack(
        [
            # Xβ⁽ʲ⁾ for each qⱼ (block diagonal matrix)
            sparse.kron(sparse.eye(num_quantiles, dtype=X.dtype), X),
            # t⁽ʲ⁾ not used in this constraint
            csr_matrix((num_quantiles * num_samples, num_quantiles * num_features), dtype=X.dtype),
            # -Δ⁽ʲ⁾⁺ for each qⱼ (block diagonal matrix)
            -sparse.eye(num_quantiles * num_samples, dtype=X.dtype),
            # Δ⁽ʲ⁾⁻ for each qⱼ (block diagonal matrix)
            sparse.eye(num_quantiles * num_samples, dtype=X.dtype),
        ]
    )
    b_eq = np.tile(y, num_quantiles)
    # Construct the inequalities -t⁽ʲ⁾ <= β⁽ʲ⁾ <= t⁽ʲ⁾ for each quantile rank qⱼ so that
    # t⁽ʲ⁾ := |β⁽ʲ⁾|. Also construct the monotonicity constraint Xβ⁽ʲ⁾ <= Xβ⁽ʲ⁺¹⁾ for each qⱼ,
    # equivalent to Δ⁽ʲ⁾⁺ - Δ⁽ʲ⁾⁻ <= Δ⁽ʲ⁺¹⁾⁺ - Δ⁽ʲ⁺¹⁾⁻.
    zeros_Δ = csr_matrix(
        (num_quantiles * num_features, 2 * num_quantiles * num_samples), dtype=X.dtype
    )
    zeros_βt = csr_matrix(
        ((num_quantiles - 1) * num_samples, 2 * num_quantiles * num_features), dtype=X.dtype
    )
    A_ub = sparse.vstack(
        [
            sparse.hstack(
                [
                    sparse.eye(num_quantiles * num_features, dtype=X.dtype),  # β⁽ʲ⁾
                    -sparse.eye(num_quantiles * num_features, dtype=X.dtype),  # -t⁽ʲ⁾
                    zeros_Δ,  # Δ⁽ʲ⁾⁺ and Δ⁽ʲ⁾⁺ not used for this constraint
                ]
            ),
            sparse.hstack(
                [
                    -sparse.eye(num_quantiles * num_features, dtype=X.dtype),  # -β⁽ʲ⁾
                    -sparse.eye(num_quantiles * num_features, dtype=X.dtype),  # -t⁽ʲ⁾
                    zeros_Δ,  # Δ⁽ʲ⁾⁺ and Δ⁽ʲ⁾⁺ not used for this constraint
                ]
            ),
            sparse.hstack(
                [
                    zeros_βt,
                    sparse.kron(
                        sparse.diags(
                            diagonals=[1, -1],  # Δ⁽ʲ⁾⁺ - Δ⁽ʲ⁺¹⁾⁺
                            offsets=[0, 1],
                            shape=(num_quantiles - 1, num_quantiles),
                            dtype=X.dtype,
                        ),
                        sparse.eye(num_samples, dtype=X.dtype),
                    ),
                    sparse.kron(
                        sparse.diags(
                            diagonals=[-1, 1],  # -Δ⁽ʲ⁾⁻ + Δ⁽ʲ⁺¹⁾⁻
                            offsets=[0, 1],
                            shape=(num_quantiles - 1, num_quantiles),
                            dtype=X.dtype,
                        ),
                        sparse.eye(num_samples, dtype=X.dtype),
                    ),
                ]
            ),
        ]
    )
    b_ub = np.zeros(A_ub.shape[0], dtype=X.dtype)
    # Construct the bounds.
    bounds = (
        ([(None, None)] * num_quantiles * num_features)  # β⁽ʲ⁾ for each qⱼ
        + ([(0, None)] * num_quantiles * num_features)  # t⁽ʲ⁾ for each qⱼ
        + ([(0, None)] * num_quantiles * num_samples)  # Δ⁽ʲ⁾⁺ for each qⱼ
        + ([(0, None)] * num_quantiles * num_samples)  # Δ⁽ʲ⁾⁻ for each qⱼ
    )
    # Solve the Coherent Quantile Regression LP.
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    # Extract the solution.
    β_full: FloatMatrix[F] = result.x[: num_quantiles * num_features].astype(y.dtype)
    β_full = β_full.reshape(num_quantiles, num_features).T
    # Drop the buffer quantile ranks we introduced earlier.
    β = β_full[:, 0 :: (coherence_buffer + 1)]
    return β, β_full


class CoherentLinearQuantileRegressor(RegressorMixin, BaseEstimator):
    """Coherent Linear Quantile Regressor.

    A linear model that regresses multiple quantiles coherently so that the predicted quantiles for
    a given example increase monotonically.
    """

    def __init__(
        self,
        *,
        quantiles: npt.ArrayLike = (0.025, 0.5, 0.975),
        fit_intercept: bool = True,
        coherence_buffer: int = 3,
    ) -> None:
        """Initialize the Coherent Quantile Regressor.

        Parameters
        ----------
        quantiles
            The target quantiles to fit and predict.
        fit_intercept
            Whether to fit an intercept term.
        coherence_buffer
            The number of auxiliary quantiles to introduce. Smaller is faster, larger yields more
            coherent quantiles.
        """
        self.quantiles = quantiles
        self.fit_intercept = fit_intercept
        self.coherence_buffer = coherence_buffer

    def fit(
        self, X: FloatMatrix[F], y: FloatVector[F], *, sample_weight: FloatVector[F] | None = None
    ) -> "CoherentLinearQuantileRegressor":
        """Fit this predictor."""
        # Validate input with backward compatibility
        if validate_data is not None:
            # For sklearn >= 1.6
            X, y = validate_data(self, X, y, y_numeric=True)
        else:
            # For sklearn < 1.6
            X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_: int = X.shape[1]
        self.y_dtype_: npt.DTypeLike = (  # Used to cast predictions to the correct dtype.
            X.dtype if np.issubdtype(y.dtype, np.integer) else y.dtype
        )
        X, y = X.astype(np.float64), y.astype(np.float64)  # To support datetime64 and timedelta64.

        # Validate sample weights - this needs to be strict to pass sklearn's checks
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.ndim != 1:
                error_msg = "Sample weights must be 1D array"
                raise ValueError(error_msg)
            if len(sample_weight) != X.shape[0]:
                error_msg = (
                    f"sample_weight.shape == {sample_weight.shape}, expected {(X.shape[0],)}"
                )
                raise ValueError(error_msg)
            check_consistent_length(y, sample_weight)
            sample_weight = sample_weight.astype(y.dtype)
        # Add a constant column to X to allow for a bias in the regression.
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        # Fit the coherent quantile regression model.
        self.β_, self.β_full_ = coherent_linear_quantile_regression(
            X,
            y,
            quantiles=np.asarray(self.quantiles),
            sample_weight=sample_weight,
            coherence_buffer=self.coherence_buffer,
        )
        return self

    def predict(self, X: FloatMatrix[F]) -> FloatMatrix[F]:
        """Predict the output on a given dataset."""
        # Check input with backward compatibility
        if validate_data is not None:
            # For sklearn >= 1.6
            X = validate_data(self, X, reset=False, dtype=np.float64)
        else:
            # For sklearn < 1.6
            X = check_array(X, dtype=np.float64)
        check_is_fitted(self)
        # Add a constant column to X to allow for a bias in the regression.
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        # Predict the output.
        ŷ: FloatMatrix[F] = X @ self.β_
        # Map back to the training target dtype.
        ŷ = np.squeeze(ŷ, axis=1 if ŷ.shape[1] == 1 else ())
        if not np.issubdtype(self.y_dtype_, np.integer):
            ŷ.astype(self.y_dtype_)
        return ŷ

    def intercept_clip(
        self, X: FloatMatrix[F], y: FloatVector[F], *, sample_weight: FloatVector[F] | None = None
    ) -> FloatMatrix[F]:
        """Compute a clip for a delta on the intercept that retains quantile coherence."""
        check_is_fitted(self)
        # Validate input with backward compatibility
        if validate_data is not None:
            # For sklearn >= 1.6
            X, y = validate_data(self, X, y, y_numeric=True)
        else:
            # For sklearn < 1.6
            X, y = check_X_y(X, y, y_numeric=True)
        X, y = X.astype(np.float64), y.astype(np.float64)

        # Add sample_weight parameter to match sklearn's API expectations
        # but we don't actually use it for this method
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.ndim != 1:
                error_msg = "Sample weights must be 1D array"
                raise ValueError(error_msg)
            if len(sample_weight) != X.shape[0]:
                error_msg = (
                    f"sample_weight.shape == {sample_weight.shape}, expected {(X.shape[0],)}"
                )
                raise ValueError(error_msg)
            check_consistent_length(y, sample_weight)
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        Q = X @ self.β_full_ - y[:, np.newaxis]
        β_intercept_clip = np.vstack(
            [
                np.insert(np.max(Q[:, :-1] - Q[:, 1:], axis=0), 0, -np.inf),
                np.append(np.min(Q[:, 1:] - Q[:, :-1], axis=0), np.inf),
            ]
        )
        β_intercept_clip[:, β_intercept_clip[0, :] >= β_intercept_clip[1, :]] = 0
        β_intercept_clip = β_intercept_clip[:, 0 :: (self.coherence_buffer + 1)]
        return β_intercept_clip
