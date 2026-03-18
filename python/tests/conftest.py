"""Shared test fixtures for pyztraj tests."""

from pathlib import Path

import numpy as np
import pytest

# Test data directory (project root / validation / test_data)
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "validation" / "test_data"


@pytest.fixture()
def pdb_path() -> Path:
    """Path to the 3tvj_I test PDB file."""
    p = TEST_DATA_DIR / "3tvj_I.pdb"
    if not p.exists():
        pytest.skip(f"Test PDB not found: {p}")
    return p


@pytest.fixture()
def xtc_path() -> Path:
    """Path to the 3tvj_I_R1 test XTC file."""
    p = TEST_DATA_DIR / "3tvj_I_R1.xtc"
    if not p.exists():
        pytest.skip(f"Test XTC not found: {p}")
    return p


@pytest.fixture()
def simple_coords() -> np.ndarray:
    """Simple 4-atom coordinates for geometry tests."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
