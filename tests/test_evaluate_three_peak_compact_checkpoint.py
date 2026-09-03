from __future__ import annotations

from scripts.scan_three_peak_weight_combinations import make_event_weights

import numpy as np


def test_three_peak_event_weights_preserve_pair_membership():
    peak_ids = np.asarray(["ba133_356kev", "na22_511kev", "cs137_662kev"])
    weights = make_event_weights(
        peak_ids,
        {"ba356": 0.4, "na511": 0.4, "cs662": 0.2},
    )
    assert weights.shape == (6,)
    np.testing.assert_allclose(weights[::2], [0.4, 0.4, 0.2])
    np.testing.assert_allclose(weights[1::2], weights[::2])
