"""Test fixtures."""

from typing import TypeAlias

import pandas as pd
import pytest
import sklearn.datasets
from _pytest.fixtures import SubRequest
from sklearn.model_selection import train_test_split

Dataset: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@pytest.fixture(
    params=[
        pytest.param(
            43926,
            id="dataset:ames_housing",  # Regression
        ),
        pytest.param(
            287,
            id="dataset:wine_quality",  # Regression
        ),
    ],
)
def dataset(request: SubRequest) -> Dataset:
    """Train and test dataset fixture."""
    # Download the dataset.
    X, y = sklearn.datasets.fetch_openml(
        data_id=request.param, return_X_y=True, as_frame=True, parser="auto"
    )
    # Split in train and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    return X_train, X_test, y_train, y_test
