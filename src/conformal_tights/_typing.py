"""Conformal Tights types."""

from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

F = TypeVar("F", np.float32, np.float64)

FloatVector: TypeAlias = npt.NDArray[F]
FloatMatrix: TypeAlias = npt.NDArray[F]
FloatTensor: TypeAlias = npt.NDArray[F]
