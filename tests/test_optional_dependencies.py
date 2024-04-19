"""Test this package's optional dependencies."""

import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("optional_dependency", ["darts", "pandas"])
def test_optional_dependencies(optional_dependency: str) -> None:
    """Test that we get an expected error when an optional dependency are not available."""
    # Prevent the optional dependency from being loaded.
    with patch.dict("sys.modules", {optional_dependency: None}):
        # Unload Conformal Tights.
        mods_to_unload = [mod for mod in sys.modules if mod.startswith("conformal_tights")]
        for mod in mods_to_unload:
            del sys.modules[mod]

        # Reload Conformal Tights now that the selected optional dependency is not available.
        from conformal_tights import ConformalCoherentQuantileRegressor, DartsForecaster

        # Test that we raise the appropriate error.
        conformal_predictor = ConformalCoherentQuantileRegressor()
        with pytest.raises(ImportError, match=f".*install.*{optional_dependency}.*"):
            _ = DartsForecaster(model=conformal_predictor)

        # Unload Conformal Tights again.
        mods_to_unload = [mod for mod in sys.modules if mod.startswith("conformal_tights")]
        for mod in mods_to_unload:
            del sys.modules[mod]
