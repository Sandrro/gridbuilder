"""Utilities for scheduled sampling and other curriculum schedules."""
from __future__ import annotations


def linear_warmup(final_prob: float, current_step: int, total_steps: int) -> float:
    """Linearly increase the scheduled sampling probability up to *final_prob*.

    Parameters
    ----------
    final_prob:
        Target probability at the end of training.
    current_step:
        Current optimization step (0-indexed).
    total_steps:
        Total number of optimization steps.
    """
    if total_steps <= 0:
        return 0.0
    progress = min(max(current_step / total_steps, 0.0), 1.0)
    return final_prob * progress
