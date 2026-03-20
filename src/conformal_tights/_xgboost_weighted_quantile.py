"""Weighted quantile matching XGBoost < 3.1 initialisation behavior."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from conformal_tights._typing import FloatVector

F = TypeVar("F", np.float32, np.float64)


def _weighted_quantile(
    y: FloatVector[F], quantiles: npt.ArrayLike, /, *, sample_weight: FloatVector[F] | None = None
) -> FloatVector[np.float64]:
    """Compute weighted quantiles using XGBoost < 3.1's initialisation formula."""
    y = np.asarray(y, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if y.size == 0:
        return np.full(quantiles.shape, np.nan, dtype=np.float64)
    if sample_weight is None:
        unweighted_quantiles: FloatVector[np.float64] = np.asarray(
            np.quantile(y, quantiles, method="weibull"), dtype=np.float64
        )
        return unweighted_quantiles
    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    sample_weight_array = np.asarray(sample_weight, dtype=np.float64)[order]
    weight_cdf = np.cumsum(sample_weight_array)
    thresholds = weight_cdf[-1] * quantiles
    indices = np.searchsorted(weight_cdf, thresholds, side="left")
    weighted_quantiles: FloatVector[np.float64] = y_sorted[indices]
    return weighted_quantiles
