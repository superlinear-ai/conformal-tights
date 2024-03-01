"""Test Conformal Tights."""

import conformal_tights


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(conformal_tights.__name__, str)
