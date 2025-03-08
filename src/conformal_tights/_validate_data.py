"""Compatibility wrapper for sklearn's data validation utilities."""

from typing import Any, Literal, TypeVar, overload

from sklearn.base import BaseEstimator

XLike = TypeVar("XLike")
YLike = TypeVar("YLike")


@overload
def validate_data(
    estimator: BaseEstimator,
    X: XLike,
    y: YLike,
    *,
    reset: bool = True,
    **check_params: Any,
) -> tuple[XLike, YLike]: ...


@overload
def validate_data(
    estimator: BaseEstimator,
    X: XLike,
    y: Literal["no_validation"] = "no_validation",
    *,
    reset: bool = True,
    **check_params: Any,
) -> XLike: ...


def validate_data(
    estimator: BaseEstimator,
    X: XLike,
    y: YLike | Literal["no_validation"] = "no_validation",
    *,
    reset: bool = True,
    **check_params: Any,
) -> XLike | tuple[XLike, YLike]:
    """Validate X and y with validate_data if available, else with check_array or check_X_y."""
    try:
        # scikit-learn >= v1.6
        from sklearn.utils.validation import validate_data as sk_validate_data

        return sk_validate_data(estimator, X=X, y=y, reset=reset, **check_params)  # type: ignore[no-any-return]

    except ImportError:
        # scikit-learn < v1.6
        from sklearn.utils.validation import check_array as sk_check_array
        from sklearn.utils.validation import check_X_y as sk_check_X_y

        check_params["estimator"] = estimator  # Include estimator name in warnings.
        if "ensure_all_finite" in check_params:
            check_params["force_all_finite"] = check_params.pop("ensure_all_finite")
        if isinstance(y, str) and y == "no_validation":
            return sk_check_array(X, **check_params)  # type: ignore[no-any-return]
        return sk_check_X_y(X, y, **check_params)  # type: ignore[no-any-return]
