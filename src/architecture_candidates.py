"""Small FPGA-oriented architecture candidates for the HPGe experiment.

These models are intentionally configurable and shape-stable. They are used for
hardware-pipeline profiling and later corrected-dataset screening; this module
does not encode scientific model-selection results.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise temporal filtering followed by pointwise channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel: int = 7) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=in_channels,
                bias=False,
            ),
            _group_norm(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.main(values) + self.skip(values))


class DSCNN(nn.Module):
    """Compact depthwise-separable 1-D CNN."""

    def __init__(self, input_channels: int = 2, width: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=11, stride=2, padding=5, bias=False),
            _group_norm(width),
            nn.GELU(),
            DepthwiseSeparableBlock(width, width, stride=1),
            DepthwiseSeparableBlock(width, 2 * width, stride=2),
            DepthwiseSeparableBlock(2 * width, 3 * width, stride=2),
            DepthwiseSeparableBlock(3 * width, 3 * width, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(3 * width, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.head(self.features(values)).squeeze(1)


class TCNBlock(nn.Module):
    """Residual dilated temporal block with a fixed receptive-field schedule."""

    def __init__(self, channels: int, dilation: int, kernel: int = 7) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False),
            _group_norm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.layers(values) + values)


class TCN(nn.Module):
    """Residual temporal convolutional network with progressive dilation."""

    def __init__(self, input_channels: int = 2, width: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=11, stride=2, padding=5, bias=False),
            _group_norm(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            TCNBlock(width, dilation=1),
            TCNBlock(width, dilation=2),
            TCNBlock(width, dilation=4),
            TCNBlock(width, dilation=8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(values))).squeeze(1)


class _RateBranch(nn.Module):
    def __init__(self, input_channels: int, width: int, slow: bool) -> None:
        super().__init__()
        self.rate_change = nn.AvgPool1d(kernel_size=4, stride=4) if slow else nn.Identity()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=9, stride=2, padding=4, bias=False),
            _group_norm(width),
            nn.GELU(),
            nn.Conv1d(width, 2 * width, kernel_size=7, stride=2, padding=3, bias=False),
            _group_norm(2 * width),
            nn.GELU(),
            nn.Conv1d(2 * width, 3 * width, kernel_size=5, stride=2, padding=2, bias=False),
            _group_norm(3 * width),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.features(self.rate_change(values))


class MultiRateHPGE(nn.Module):
    """Fast full-rate and slow downsampled branches with late fusion."""

    def __init__(self, input_channels: int = 2, width: int = 16) -> None:
        super().__init__()
        self.fast = _RateBranch(input_channels, width, slow=False)
        self.slow = _RateBranch(input_channels, width, slow=True)
        self.head = nn.Sequential(
            nn.Linear(6 * width, 3 * width),
            nn.GELU(),
            nn.Linear(3 * width, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.head(torch.cat((self.fast(values), self.slow(values)), dim=1)).squeeze(1)


class CNNGRU(nn.Module):
    """Convolutional reduction followed by a small recurrent temporal state."""

    def __init__(self, input_channels: int = 2, width: int = 24, hidden_size: int = 24) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=11, stride=2, padding=5, bias=False),
            _group_norm(width),
            nn.GELU(),
            nn.Conv1d(width, 2 * width, kernel_size=7, stride=2, padding=3, bias=False),
            _group_norm(2 * width),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=2 * width,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values: Tensor) -> Tensor:
        sequence = self.convolution(values).transpose(1, 2)
        sequence, _ = self.gru(sequence)
        return self.head(sequence[:, -1]).squeeze(1)


def build_candidate(name: str, input_channels: int = 2, width: int = 24) -> nn.Module:
    """Construct one named candidate for profiling or later training."""

    if name == "ds_cnn":
        return DSCNN(input_channels=input_channels, width=width)
    if name == "tcn":
        return TCN(input_channels=input_channels, width=width)
    if name == "multi_rate_hpge":
        return MultiRateHPGE(input_channels=input_channels, width=max(8, width // 2))
    if name == "cnn_gru":
        return CNNGRU(input_channels=input_channels, width=width, hidden_size=width)
    raise ValueError(f"Unknown architecture candidate: {name}")
