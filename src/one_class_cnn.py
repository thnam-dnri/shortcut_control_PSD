"""Collapse-monitored one-class Compact CNN components."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def group_norm_no_affine(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels, affine=False)


class OneClassResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                5,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            group_norm_no_affine(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                5,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            group_norm_no_affine(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.main(values) + self.skip(values))


class OneClassCompactEncoder(nn.Module):
    """Compact feature extractor with an eight-dimensional bias-free projection."""

    def __init__(self, input_channels: int = 2, width: int = 24, embedding_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, width, 11, stride=2, padding=5, bias=False),
            group_norm_no_affine(width),
            nn.GELU(),
            OneClassResidualBlock(width, width, 1, 1),
            OneClassResidualBlock(width, 2 * width, 2, 1),
            OneClassResidualBlock(2 * width, 3 * width, 2, 2),
            OneClassResidualBlock(3 * width, 3 * width, 2, 4),
        )
        self.projection = nn.Linear(6 * width, embedding_dim, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        features = self.features(values)
        pooled = torch.cat(
            (torch.mean(features, dim=2), torch.amax(features, dim=2)), dim=1
        )
        return self.projection(pooled)


def clamp_center(center: Tensor, epsilon: float = 0.1) -> Tensor:
    result = center.clone()
    small = torch.abs(result) < epsilon
    result[small & (result < 0)] = -epsilon
    result[small & (result >= 0)] = epsilon
    return result


def embedding_diagnostics(embeddings: np.ndarray) -> dict[str, float | list[float]]:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Expected at least two embedding rows")
    stds = np.std(embeddings, axis=0, dtype=np.float64)
    covariance = np.cov(embeddings, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denominator = float(np.sum(np.square(eigenvalues)))
    effective_rank = (
        float(np.square(np.sum(eigenvalues)) / denominator) if denominator > 0 else 0.0
    )
    return {
        "mean_dimension_std": float(np.mean(stds)),
        "minimum_dimension_std": float(np.min(stds)),
        "effective_rank": effective_rank,
        "dimension_stds": stds.tolist(),
    }
