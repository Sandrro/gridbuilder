"""Training script for the autoregressive grid Transformer."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset import EDGE_TOKENS, GridDataset, collate_zone_batch
from models.transformer import AutoregressiveTransformer, ModelConfig
from utils import metrics as metrics_utils
from utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True, help="Path to descriptions.parquet")
    parser.add_argument("--grid", required=True, help="Path to grid_cells.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--cell-size", type=float, default=15.0, help="Cell size in meters (metadata)")
    parser.add_argument("--min-cells", type=int, default=16, help="Minimum number of cells per zone")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=4096, help="Maximum number of cells per zone")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sched-sampling", type=float, default=0.0, help="Final scheduled sampling prob")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(dataset: GridDataset, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    zone_to_type = [dataset.zone_type_to_id[str(dataset.zone_meta[zone_id].zone_type)] for zone_id in dataset.zone_ids]
    indices_by_type: Dict[int, List[int]] = {}
    for idx, zone_type in enumerate(zone_to_type):
        indices_by_type.setdefault(zone_type, []).append(idx)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for zone_type, indices in indices_by_type.items():
        indices = indices.copy()
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else max(int(len(indices) * val_fraction), 0)
        val_subset = indices[:val_count]
        train_subset = indices[val_count:]
        if not train_subset:
            train_subset = val_subset
            val_subset = []
        train_indices.extend(train_subset)
        val_indices.extend(val_subset)
    return train_indices, val_indices


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    dataset: GridDataset,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = batch["sequence_mask"]

    cell_targets = batch["cell_class"]
    cell_logits = outputs["cell_class_logits"]
    cell_loss = F.cross_entropy(cell_logits.permute(0, 2, 1), cell_targets.clamp(min=0), reduction="none")
    cell_loss = (cell_loss * mask).sum() / mask.sum().clamp_min(1.0)

    living_logits = outputs["is_living_logits"].squeeze(-1)
    living_targets = batch["is_living"]
    living_mask = batch["is_living_mask"]
    living_loss = F.binary_cross_entropy_with_logits(living_logits, living_targets, reduction="none")
    living_loss = (living_loss * living_mask).sum() / living_mask.sum().clamp_min(1.0)

    storeys_pred = outputs["storeys"].squeeze(-1)
    storeys_target = batch["storeys"]
    storeys_mask = batch["storeys_mask"]
    storeys_loss = F.smooth_l1_loss(storeys_pred, storeys_target, reduction="none")
    storeys_loss = (storeys_loss * storeys_mask).sum() / storeys_mask.sum().clamp_min(1.0)

    living_area_pred = F.softplus(outputs["living_area"].squeeze(-1))
    living_area_target = batch["living_area"]
    living_area_mask = batch["living_area_mask"]
    living_area_loss = F.smooth_l1_loss(living_area_pred, living_area_target, reduction="none")
    living_area_loss = (living_area_loss * living_area_mask).sum() / living_area_mask.sum().clamp_min(1.0)

    service_logits = outputs["service_type_logits"].permute(0, 2, 1)
    service_target = batch["service_type"]
    service_mask = batch["service_type_mask"]
    masked_target = service_target.clamp(min=0)
    service_loss = F.cross_entropy(service_logits, masked_target, reduction="none")
    service_loss = (service_loss * service_mask).sum() / service_mask.sum().clamp_min(1.0)

    service_capacity_pred = F.softplus(outputs["service_capacity"].squeeze(-1))
    service_capacity_target = batch["service_capacity"]
    service_capacity_mask = batch["service_capacity_mask"]
    service_capacity_loss = F.smooth_l1_loss(
        service_capacity_pred,
        service_capacity_target,
        reduction="none",
    )
    service_capacity_loss = (service_capacity_loss * service_capacity_mask).sum() / service_capacity_mask.sum().clamp_min(1.0)

    batch_cells = batch["num_cells"].clamp_min(1.0)
    total_living_pred = (living_area_pred * living_area_mask).sum(dim=1)
    total_living_target = (living_area_target * living_area_mask).sum(dim=1)
    pred_density = torch.log1p(total_living_pred / batch_cells)
    target_density = torch.log1p(total_living_target / batch_cells)
    living_norm = (pred_density - dataset.living_mean) / max(dataset.living_std, 1e-6)
    target_norm = (target_density - dataset.living_mean) / max(dataset.living_std, 1e-6)
    aggregate_living_loss = F.smooth_l1_loss(living_norm, target_norm)

    service_aggregate_losses: List[torch.Tensor] = []
    for idx, service in enumerate(dataset.service_types):
        type_mask = (service_target == idx).float()
        pred_sum = (service_capacity_pred * type_mask).sum(dim=1)
        target_sum = (service_capacity_target * type_mask).sum(dim=1)
        pred_density = torch.log1p(pred_sum / batch_cells)
        target_density = torch.log1p(target_sum / batch_cells)
        mean = dataset.capacity_mean.get(service, 0.0)
        std = max(dataset.capacity_std.get(service, 1.0), 1e-6)
        norm_pred = (pred_density - mean) / std
        norm_target = (target_density - mean) / std
        budget_mask = batch["service_prompt_mask"][:, idx]
        loss = F.smooth_l1_loss(norm_pred, norm_target, reduction="none")
        service_aggregate_losses.append((loss * budget_mask).sum() / budget_mask.sum().clamp_min(1.0))
    if service_aggregate_losses:
        aggregate_service_loss = torch.stack(service_aggregate_losses).mean()
    else:
        aggregate_service_loss = torch.tensor(0.0, device=mask.device)

    total_loss = (
        cell_loss
        + living_loss
        + storeys_loss
        + living_area_loss
        + service_loss
        + service_capacity_loss
        + 0.1 * (aggregate_living_loss + aggregate_service_loss)
    )

    metrics = {
        "loss/total": float(total_loss.item()),
        "loss/cell_class": float(cell_loss.item()),
        "loss/is_living": float(living_loss.item()),
        "loss/storeys": float(storeys_loss.item()),
        "loss/living_area": float(living_area_loss.item()),
        "loss/service_type": float(service_loss.item()),
        "loss/service_capacity": float(service_capacity_loss.item()),
        "loss/aggregate_living": float(aggregate_living_loss.item()),
        "loss/aggregate_service": float(aggregate_service_loss.item()),
    }
    return total_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    out_dir: Path,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    ensure_dir(out_dir / "checkpoints")
    torch.save(checkpoint, out_dir / "checkpoints" / f"epoch_{epoch}.pt")



def train_one_epoch(
    model: AutoregressiveTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    dataset: GridDataset,
    device: torch.device,
    epoch: int,
    total_steps: int,
    args: argparse.Namespace,
    global_step: int,
) -> int:
    model.train()
    logger = metrics_utils.MetricsLogger()
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                batch["cell_class"],
                batch["sequence_mask"],
                zone_type_ids=batch["zone_type_ids"],
                living_prompt=batch["living_prompt"],
                living_prompt_mask=batch["living_prompt_mask"],
                service_prompt=batch["service_prompt"],
                service_prompt_mask=batch["service_prompt_mask"],
                edge_distances=batch["edge_distances"],
            )
            loss, metrics = compute_losses(outputs, batch, dataset)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for key, value in metrics.items():
            logger.update(key, value)
        global_step += 1
        if (step + 1) % args.log_every == 0:
            averages = logger.to_dict()
            log_items = " ".join(f"{k}={v:.4f}" for k, v in sorted(averages.items()))
            print(f"Epoch {epoch} step {step+1}: {log_items}")
            logger = metrics_utils.MetricsLogger()
    return global_step


def evaluate(
    model: AutoregressiveTransformer,
    loader: DataLoader,
    dataset: GridDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    logger = metrics_utils.MetricsLogger()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                batch["cell_class"],
                batch["sequence_mask"],
                zone_type_ids=batch["zone_type_ids"],
                living_prompt=batch["living_prompt"],
                living_prompt_mask=batch["living_prompt_mask"],
                service_prompt=batch["service_prompt"],
                service_prompt_mask=batch["service_prompt_mask"],
                edge_distances=batch["edge_distances"],
            )
            loss, metrics = compute_losses(outputs, batch, dataset)
            logger.update("loss/total", float(loss.item()))
            for key, value in metrics.items():
                logger.update(f"val_{key}", value)
    return logger.to_dict()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(args.out_dir)

    dataset = GridDataset(
        args.grid,
        args.descriptions,
        min_cells=args.min_cells,
        max_context=args.max_context,
    )

    train_indices, val_indices = stratified_split(dataset, args.val_split, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_zone_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_zone_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_positions=args.max_context,
    )
    model = AutoregressiveTransformer(
        config,
        service_vocab=len(dataset.service_types),
        zone_vocab=len(dataset.zone_type_to_id),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    total_steps = max(len(train_loader) * args.epochs, 1)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            dataset,
            device,
            epoch,
            total_steps,
            args,
            global_step,
        )
        metrics = evaluate(model, val_loader, dataset, device)
        if metrics:
            log_items = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            print(f"Epoch {epoch} validation: {log_items}")
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, out_dir)

    torch.save(model.state_dict(), out_dir / "model.pt")

    vocab = {
        "service_types": dataset.service_types,
        "zone_types": [dataset.id_to_zone_type[i] for i in sorted(dataset.id_to_zone_type)],
        "edge_tokens": EDGE_TOKENS,
    }
    save_json(vocab, out_dir / "vocab.json")

    norm_stats = {
        "living_mean": dataset.living_mean,
        "living_std": dataset.living_std,
        "service_capacity_mean": dataset.capacity_mean,
        "service_capacity_std": dataset.capacity_std,
        "cell_size_m": args.cell_size,
    }
    save_json(norm_stats, out_dir / "norm_stats.json")

    inference_config = {
        "model": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
            "max_positions": config.max_positions,
        },
        "service_vocab": len(dataset.service_types),
        "zone_vocab": len(dataset.zone_type_to_id),
        "cell_size": args.cell_size,
    }
    save_json(inference_config, out_dir / "inference_config.json")


if __name__ == "__main__":
    main()
