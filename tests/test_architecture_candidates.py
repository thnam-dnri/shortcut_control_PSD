from __future__ import annotations

import pytest
import torch

from src.architecture_candidates import build_candidate


@pytest.mark.parametrize("name", ["ds_cnn", "tcn", "multi_rate_hpge", "cnn_gru"])
def test_candidate_forward_shape_and_finiteness(name: str) -> None:
    model = build_candidate(name, input_channels=2, width=8)
    values = torch.randn(2, 2, 750)
    with torch.no_grad():
        output = model(values)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
