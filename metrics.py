"""Regression metrics utilities for torch tensors."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["mae", "rmse", "r2", "mape"]


def _prepare_tensors(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened prediction, target, and mask tensors."""

    y_pred = torch.nan_to_num(y_pred).to(dtype=torch.float32)
    y_true = torch.nan_to_num(y_true).to(dtype=torch.float32)

    device = y_true.device
    if mask is None:
        mask_tensor = torch.ones_like(y_true, dtype=torch.bool, device=device)
    else:
        mask_tensor = mask.to(device=device)
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor != 0
        mask_tensor = mask_tensor & torch.ones_like(y_true, dtype=torch.bool, device=device)

    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    mask_flat = mask_tensor.reshape(-1)
    return y_pred_flat, y_true_flat, mask_flat


def _empty_result(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def mae(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean absolute error of *y_pred* and *y_true* using *mask* to filter entries."""

    y_pred_flat, y_true_flat, mask_flat = _prepare_tensors(y_pred, y_true, mask)
    count = mask_flat.sum()
    if count.item() == 0:
        return _empty_result(y_true_flat.device)
    diff = torch.abs(y_pred_flat - y_true_flat)
    return diff.masked_select(mask_flat).mean()


def rmse(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Root mean squared error of *y_pred* and *y_true* using *mask* to filter entries."""

    y_pred_flat, y_true_flat, mask_flat = _prepare_tensors(y_pred, y_true, mask)
    count = mask_flat.sum()
    if count.item() == 0:
        return _empty_result(y_true_flat.device)
    diff = y_pred_flat - y_true_flat
    mse = diff.pow(2).masked_select(mask_flat).mean()
    return torch.sqrt(mse)


def r2(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Coefficient of determination (R^2) between *y_pred* and *y_true*."""

    y_pred_flat, y_true_flat, mask_flat = _prepare_tensors(y_pred, y_true, mask)
    valid_pred = y_pred_flat.masked_select(mask_flat)
    valid_true = y_true_flat.masked_select(mask_flat)
    if valid_true.numel() == 0:
        return _empty_result(y_true_flat.device)
    mean_true = valid_true.mean()
    ss_tot = torch.sum((valid_true - mean_true) ** 2)
    if ss_tot.item() <= 0:
        return _empty_result(y_true_flat.device)
    ss_res = torch.sum((valid_true - valid_pred) ** 2)
    return 1.0 - ss_res / ss_tot.clamp_min(1e-12)


def mape(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean absolute percentage error with denominator clamped by *eps*."""

    y_pred_flat, y_true_flat, mask_flat = _prepare_tensors(y_pred, y_true, mask)
    count = mask_flat.sum()
    if count.item() == 0:
        return _empty_result(y_true_flat.device)
    denom = torch.clamp(y_true_flat.abs(), min=eps)
    ape = torch.abs(y_pred_flat - y_true_flat) / denom
    return ape.masked_select(mask_flat).mean()
