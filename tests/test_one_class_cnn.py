from __future__ import annotations

import numpy as np
import torch

from src.one_class_cnn import (
    OneClassCompactEncoder,
    clamp_center,
    embedding_diagnostics,
)


def test_one_class_encoder_shape_and_bias_contract():
    model = OneClassCompactEncoder(input_channels=2, width=8, embedding_dim=4)
    values = torch.randn(3, 2, 750)
    embeddings = model(values)
    assert embeddings.shape == (3, 4)
    assert model.projection.bias is None
    assert all(
        not module.affine
        for module in model.modules()
        if isinstance(module, torch.nn.GroupNorm)
    )


def test_center_clamp_and_embedding_diagnostics():
    center = clamp_center(torch.tensor([-0.01, 0.0, 0.4]), epsilon=0.1)
    torch.testing.assert_close(center, torch.tensor([-0.1, 0.1, 0.4]))
    embeddings = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    diagnostics = embedding_diagnostics(embeddings)
    assert diagnostics["effective_rank"] == 2.0
    assert diagnostics["mean_dimension_std"] == 0.5
