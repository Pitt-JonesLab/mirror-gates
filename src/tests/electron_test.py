"""Test the Electron class."""

from src.virtual_swap.main import Electron


def test_electron():
    """Test basic Electron functionality."""
    electron = Electron()
    assert electron.charge == -1
    assert electron.mass == 2.1
    assert electron.compute_momentum(2.1) == 4.41


def test_electron_with_name():
    """Test the Electron class with a name."""
    electron = Electron(name="Electron")
    assert electron.name == "Electron"
