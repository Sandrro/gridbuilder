"""Utility metrics tracking helpers used during training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class AverageMeter:
    """Keeps a running average of a scalar quantity."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class MetricsLogger:
    """Accumulates training metrics for logging."""

    meters: Dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self.meters:
            self.meters[key] = AverageMeter()
        self.meters[key].update(value, n)

    def to_dict(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error that ignores entries where *mask* == 0."""
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error that ignores entries where *mask* == 0."""
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
