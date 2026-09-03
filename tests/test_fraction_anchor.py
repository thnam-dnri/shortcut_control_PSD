from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import anchor_positions, fraction_anchor, t10_anchor


def synthetic_charge() -> np.ndarray:
    charge = np.zeros((2, 4500), dtype=np.float32)
    charge[0, 1000:1101] = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    charge[0, 1101:] = 100.0
    charge[1, 1000:1201] = np.linspace(0.0, 200.0, 201, dtype=np.float32)
    charge[1, 1201:] = 200.0
    return charge


@pytest.mark.parametrize(
    ("fraction", "expected"),
    (
        (0.1, np.asarray([1010, 1020])),
        (0.5, np.asarray([1050, 1100])),
        (0.9, np.asarray([1090, 1180])),
    ),
)
def test_fraction_anchor_first_rising_crossing(
    fraction: float, expected: np.ndarray
) -> None:
    positions, fallbacks = fraction_anchor(synthetic_charge(), fraction)
    np.testing.assert_array_equal(positions, expected)
    assert fallbacks == 0


def test_t10_wrapper_preserves_fraction_definition() -> None:
    charge = synthetic_charge()
    direct, direct_fallbacks = fraction_anchor(charge, 0.1)
    wrapped, wrapped_fallbacks = t10_anchor(charge)
    np.testing.assert_array_equal(wrapped, direct)
    assert wrapped_fallbacks == direct_fallbacks


@pytest.mark.parametrize(("anchor", "expected"), (("t50", 1050), ("t90", 1090)))
def test_anchor_positions_supports_fraction_anchors(anchor: str, expected: int) -> None:
    charge = synthetic_charge()[:1]
    current = np.gradient(charge, axis=1).astype(np.float32)
    positions, fallbacks = anchor_positions(charge, current, anchor)
    assert positions.tolist() == [expected]
    assert fallbacks == 0


@pytest.mark.parametrize("fraction", (0.0, -0.1, 1.1))
def test_fraction_anchor_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        fraction_anchor(synthetic_charge(), fraction)
