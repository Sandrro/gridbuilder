"""Training script for the autoregressive grid Transformer."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.dataset import EDGE_TOKENS, GridDataset, collate_zone_batch
from models.transformer import AutoregressiveTransformer, ModelConfig
from utils import metrics as metrics_utils
from utils.io import ensure_dir, save_json
from utils.scheduling import linear_warmup
from torch.distributions import Dirichlet


PROMPT_WEIGHT_START = 0.1
PROMPT_WEIGHT_END = 0.5
PROMPT_JITTER_PROB = 0.4
HARD_SCENARIO_PROB = 0.5

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument(
        "--tensorboard-logdir",
        default="tensorboard",
        help="Relative directory inside --out-dir for TensorBoard logs",
    )
    parser.add_argument("--upload-to-hf", action="store_true", help="Upload artifacts to HuggingFace Hub")
    parser.add_argument("--hf-repo-id", help="Target HuggingFace Hub repository id")
    parser.add_argument("--hf-token", help="HuggingFace token; falls back to cached token when omitted")
    parser.add_argument("--hf-branch", default="main", help="Branch to upload artifacts to")
    parser.add_argument("--hf-repo-path", default="", help="Optional path inside the HuggingFace repository")
    parser.add_argument("--hf-private", action="store_true", help="Create private repository when allowed")
    parser.add_argument("--hf-allow-create", action="store_true", help="Create the repository if it does not exist")
    parser.add_argument(
        "--hf-commit-message",
        default="Add GridBuilder training artifacts",
        help="Commit message used for the HuggingFace upload",
    )
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
    *,
    lambda_prompt: float,
    lambda_target: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = batch["sequence_mask"]

    cell_targets = batch["cell_class"]
    cell_logits = outputs["cell_class_logits"]
    cell_loss = F.cross_entropy(cell_logits.permute(0, 2, 1), cell_targets.clamp(min=0), reduction="none")
    cell_loss = (cell_loss * mask).sum() / mask.sum().clamp_min(1.0)

    living_logits = outputs["is_living_logits"].squeeze(-1)
    living_targets = batch["is_living"]
    living_mask = batch["is_living_mask"]
    building_mask = ((cell_targets == 1) | (cell_targets == 3)).float()
    service_cell_mask = ((cell_targets == 2) | (cell_targets == 3)).float()

    living_logits = living_logits * building_mask
    living_loss = F.binary_cross_entropy_with_logits(living_logits, living_targets, reduction="none")
    living_loss = (living_loss * living_mask).sum() / living_mask.sum().clamp_min(1.0)

    storeys_pred = F.softplus(outputs["storeys"].squeeze(-1)) * building_mask
    storeys_target = batch["storeys"]
    storeys_mask = batch["storeys_mask"]
    storeys_loss = F.smooth_l1_loss(storeys_pred, storeys_target, reduction="none")
    storeys_loss = (storeys_loss * storeys_mask).sum() / storeys_mask.sum().clamp_min(1.0)

    living_area_pred = F.softplus(outputs["living_area"].squeeze(-1)) * building_mask
    living_area_target = batch["living_area"]
    living_area_mask = batch["living_area_mask"]
    living_area_loss = F.smooth_l1_loss(living_area_pred, living_area_target, reduction="none")
    living_area_loss = (living_area_loss * living_area_mask).sum() / living_area_mask.sum().clamp_min(1.0)

    service_logits = outputs["service_type_logits"] * service_cell_mask.unsqueeze(-1)
    service_presence = batch["service_presence"]
    service_presence_mask = batch["service_presence_mask"]
    if service_logits.size(-1) > 0 and service_presence_mask.sum() > 0:
        service_loss_tensor = F.binary_cross_entropy_with_logits(
            service_logits,
            service_presence,
            reduction="none",
        )
        service_loss = (service_loss_tensor * service_presence_mask).sum() / service_presence_mask.sum().clamp_min(1.0)
    else:
        service_loss = torch.tensor(0.0, device=mask.device)

    service_capacity_pred = F.softplus(outputs["service_capacity"]) * service_cell_mask.unsqueeze(-1)
    service_capacity_target = batch["service_capacity"]
    service_capacity_mask = batch["service_capacity_mask"]
    if service_capacity_pred.size(-1) > 0 and service_capacity_mask.sum() > 0:
        service_capacity_loss_tensor = F.smooth_l1_loss(
            service_capacity_pred,
            service_capacity_target,
            reduction="none",
        )
        service_capacity_loss = (
            service_capacity_loss_tensor * service_capacity_mask
        ).sum() / service_capacity_mask.sum().clamp_min(1.0)
    else:
        service_capacity_loss = torch.tensor(0.0, device=mask.device)

    batch_cells = batch["num_cells"].clamp_min(1.0)
    total_living_pred = (living_area_pred * living_area_mask).sum(dim=1)
    total_living_target = (living_area_target * living_area_mask).sum(dim=1)
    pred_density = torch.log1p(total_living_pred / batch_cells)
    target_density = torch.log1p(total_living_target / batch_cells)
    living_norm = (pred_density - dataset.living_mean) / max(dataset.living_std, 1e-6)
    target_norm = (target_density - dataset.living_mean) / max(dataset.living_std, 1e-6)
    living_prompt_norm = batch["living_prompt"].squeeze(-1)
    living_prompt_mask = batch["living_prompt_mask"].squeeze(-1)

    living_target_loss = F.smooth_l1_loss(living_norm, target_norm, reduction="none")
    living_target_loss = living_target_loss.mean() if living_target_loss.ndim == 1 else living_target_loss

    living_prompt_loss = F.smooth_l1_loss(living_norm, living_prompt_norm, reduction="none")
    living_prompt_loss = (living_prompt_loss * living_prompt_mask).sum() / living_prompt_mask.sum().clamp_min(1.0)
    aggregate_living_loss = lambda_target * living_target_loss + lambda_prompt * living_prompt_loss

    service_aggregate_losses: List[torch.Tensor] = []
    service_target_losses: List[torch.Tensor] = []
    service_prompt_losses: List[torch.Tensor] = []
    for idx, service in enumerate(dataset.service_types):
        type_mask = service_capacity_mask[:, :, idx]
        pred_sum = (service_capacity_pred[:, :, idx] * type_mask).sum(dim=1)
        target_sum = (service_capacity_target[:, :, idx] * type_mask).sum(dim=1)
        pred_density = torch.log1p(pred_sum / batch_cells)
        target_density = torch.log1p(target_sum / batch_cells)
        mean = dataset.capacity_mean.get(service, 0.0)
        std = max(dataset.capacity_std.get(service, 1.0), 1e-6)
        norm_pred = (pred_density - mean) / std
        norm_target = (target_density - mean) / std
        budget_mask = batch["service_prompt_mask"][:, idx]
        target_loss = F.smooth_l1_loss(norm_pred, norm_target, reduction="none")
        target_valid = (type_mask.sum(dim=1) > 0).float()
        if target_valid.sum() > 0:
            target_loss = (target_loss * target_valid).sum() / target_valid.sum().clamp_min(1.0)
        else:
            target_loss = torch.tensor(0.0, device=mask.device)
        prompt_loss = F.smooth_l1_loss(norm_pred, batch["service_prompt"][:, idx], reduction="none")
        if budget_mask.sum() > 0:
            prompt_loss = (prompt_loss * budget_mask).sum() / budget_mask.sum().clamp_min(1.0)
        else:
            prompt_loss = torch.tensor(0.0, device=mask.device)
        combined = lambda_target * target_loss + lambda_prompt * prompt_loss
        service_aggregate_losses.append(combined)
        service_target_losses.append(target_loss)
        service_prompt_losses.append(prompt_loss)
    if service_aggregate_losses:
        aggregate_service_loss = torch.stack(service_aggregate_losses).mean()
        aggregate_service_target = torch.stack(service_target_losses).mean()
        aggregate_service_prompt = torch.stack(service_prompt_losses).mean()
    else:
        aggregate_service_loss = torch.tensor(0.0, device=mask.device)
        aggregate_service_target = torch.tensor(0.0, device=mask.device)
        aggregate_service_prompt = torch.tensor(0.0, device=mask.device)

    aggregate_living_target = living_target_loss if torch.is_tensor(living_target_loss) else torch.tensor(float(living_target_loss), device=mask.device)
    aggregate_living_prompt = living_prompt_loss if torch.is_tensor(living_prompt_loss) else torch.tensor(float(living_prompt_loss), device=mask.device)

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
        "loss/service_presence": float(service_loss.item()),
        "loss/service_capacity": float(service_capacity_loss.item()),
        "loss/aggregate_living": float(aggregate_living_loss.item()),
        "loss/aggregate_service": float(aggregate_service_loss.item()),
        "loss/aggregate_living_target": float(aggregate_living_target.item()),
        "loss/aggregate_living_prompt": float(aggregate_living_prompt.item()),
        "loss/aggregate_service_target": float(aggregate_service_target.item()),
        "loss/aggregate_service_prompt": float(aggregate_service_prompt.item()),
    }
    return total_loss, metrics


def compute_aggregate_weights(current_step: int, total_steps: int) -> Tuple[float, float]:
    if total_steps <= 1:
        return PROMPT_WEIGHT_START, 1.0 - PROMPT_WEIGHT_START
    denom = max(total_steps - 1, 1)
    progress = min(max(current_step / denom, 0.0), 1.0)
    lambda_prompt = PROMPT_WEIGHT_START + (PROMPT_WEIGHT_END - PROMPT_WEIGHT_START) * progress
    lambda_prompt = float(min(max(lambda_prompt, 0.0), 1.0))
    lambda_target = float(max(1.0 - lambda_prompt, 0.0))
    return lambda_prompt, lambda_target


def _apply_hard_service_scenario(
    totals: torch.Tensor,
    active_indices: torch.Tensor,
    dataset: GridDataset,
) -> torch.Tensor:
    adjusted = totals.clone()
    if adjusted.numel() == 0:
        return adjusted
    index_map = {int(idx): pos for pos, idx in enumerate(active_indices.tolist())}
    park_idx = dataset.service_type_to_id.get("Park")
    playground_idx = dataset.service_type_to_id.get("Playground")
    if park_idx is not None and park_idx in index_map:
        adjusted[index_map[park_idx]] = 0.0
    if playground_idx is not None and playground_idx in index_map:
        adjusted[index_map[playground_idx]] = adjusted[index_map[playground_idx]] * 2.0
    return adjusted


def apply_prompt_jitter(batch: Dict[str, torch.Tensor], dataset: GridDataset, device: torch.device) -> None:
    if PROMPT_JITTER_PROB <= 0.0:
        return
    living_prompts = batch["living_prompt"]
    batch_size = living_prompts.size(0)
    if batch_size == 0:
        return
    jitter_flags = torch.rand(batch_size, device=device) < PROMPT_JITTER_PROB
    if not jitter_flags.any():
        return
    living_std = max(dataset.living_std, 1e-6)
    living_mean = dataset.living_mean
    service_means = torch.tensor(
        [dataset.capacity_mean.get(service, 0.0) for service in dataset.service_types],
        device=device,
        dtype=torch.float32,
    )
    service_stds = torch.tensor(
        [max(dataset.capacity_std.get(service, 1.0), 1e-6) for service in dataset.service_types],
        device=device,
        dtype=torch.float32,
    )
    for batch_idx in torch.nonzero(jitter_flags, as_tuple=False).squeeze(-1).tolist():
        num_cells = batch["num_cells"][batch_idx].clamp_min(1.0)
        # Living area jitter
        if batch["living_prompt_mask"][batch_idx, 0] > 0:
            living_norm = living_prompts[batch_idx, 0]
            density = living_norm * living_std + living_mean
            total_living = torch.expm1(density) * num_cells
            if torch.rand(1, device=device).item() < HARD_SCENARIO_PROB:
                total_living = total_living * 1.3
            else:
                noise = torch.empty(1, device=device).uniform_(0.8, 1.2)
                total_living = total_living * noise.squeeze()
            new_density = torch.log1p(total_living / num_cells)
            living_prompts[batch_idx, 0] = (new_density - living_mean) / living_std

        # Service capacity jitter
        service_mask = batch["service_prompt_mask"][batch_idx].bool()
        if not service_mask.any():
            continue
        service_norm = batch["service_prompt"][batch_idx]
        densities = service_norm[service_mask] * service_stds[service_mask] + service_means[service_mask]
        totals = torch.expm1(densities) * num_cells
        active_indices = torch.nonzero(service_mask, as_tuple=False).squeeze(-1)
        if torch.rand(1, device=device).item() < HARD_SCENARIO_PROB:
            totals = _apply_hard_service_scenario(totals, active_indices, dataset)
        else:
            dirichlet = Dirichlet(torch.ones_like(totals))
            proportions = dirichlet.sample().to(device)
            total_budget = totals.sum()
            if total_budget <= 1e-6:
                totals = totals * 0.0
            else:
                totals = proportions * total_budget
        new_densities = torch.log1p(totals / num_cells)
        normalized = (new_densities - service_means[service_mask]) / service_stds[service_mask]
        service_norm[service_mask] = normalized
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


def create_summary_writer(out_dir: Path, args: argparse.Namespace) -> Optional["SummaryWriter"]:
    if not args.tensorboard_logdir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        logging.warning("TensorBoard is not installed; skipping SummaryWriter setup")
        return None
    log_dir = ensure_dir(out_dir / args.tensorboard_logdir)
    logging.info("TensorBoard logs will be written to %s", log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def upload_artifacts_to_hf(out_dir: Path, args: argparse.Namespace) -> None:
    if not args.upload_to_hf:
        return
    if not args.hf_repo_id:
        raise ValueError("--hf-repo-id must be provided when --upload-to-hf is used")
    try:
        from huggingface_hub import HfApi, HfFolder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required for uploading to HuggingFace") from exc

    token = args.hf_token or HfFolder.get_token()
    if not token:
        raise RuntimeError("HuggingFace token not found; provide --hf-token or login with huggingface-cli")

    api = HfApi()
    if args.hf_allow_create:
        logging.info("Ensuring HuggingFace repository %s exists", args.hf_repo_id)
        api.create_repo(
            repo_id=args.hf_repo_id,
            repo_type="model",
            token=token,
            private=args.hf_private,
            exist_ok=True,
        )

    revision = args.hf_branch or "main"
    logging.info(
        "Uploading training artifacts from %s to HuggingFace Hub repo %s (branch %s)",
        out_dir,
        args.hf_repo_id,
        revision,
    )
    try:
        api.upload_folder(
            repo_id=args.hf_repo_id,
            repo_type="model",
            folder_path=str(out_dir),
            path_in_repo=args.hf_repo_path or "",
            token=token,
            commit_message=args.hf_commit_message,
            revision=revision,
        )
    except Exception as exc:  # pragma: no cover - network errors are environment specific
        raise RuntimeError(f"Failed to upload artifacts to HuggingFace Hub: {exc}") from exc
    logging.info("Upload to HuggingFace Hub completed successfully")



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
    writer: Optional["SummaryWriter"],
) -> int:
    model.train()
    logger = metrics_utils.MetricsLogger()
    progress = tqdm(loader, desc=f"Train {epoch}", leave=False, total=len(loader))
    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        apply_prompt_jitter(batch, dataset, device)
        schedule_total = max(total_steps - 1, 1)
        current_step = min(global_step, schedule_total)
        sampling_prob = linear_warmup(args.sched_sampling, current_step, schedule_total)
        forced_prev_tokens: Optional[torch.Tensor] = None
        if sampling_prob > 0.0:
            with torch.no_grad():
                teacher_outputs = model(
                    batch["cell_class"],
                    batch["sequence_mask"],
                    zone_type_ids=batch["zone_type_ids"],
                    living_prompt=batch["living_prompt"],
                    living_prompt_mask=batch["living_prompt_mask"],
                    service_prompt=batch["service_prompt"],
                    service_prompt_mask=batch["service_prompt_mask"],
                    edge_distances=batch["edge_distances"],
                    cell_coords=batch["cell_coords"],
                )
            predicted_classes = teacher_outputs["cell_class_logits"].argmax(dim=-1)
            start_ids = torch.full(
                (batch["cell_class"].size(0), 1), 4, dtype=torch.long, device=device
            )
            teacher_prev = batch["cell_class"].clamp(min=0)
            if teacher_prev.size(1) > 0:
                forced_prev_tokens = torch.cat([start_ids, teacher_prev[:, :-1].clamp(min=0)], dim=1)
                if teacher_prev.size(1) > 1:
                    rand_mask = (
                        torch.rand(teacher_prev.size(0), teacher_prev.size(1) - 1, device=device)
                        < sampling_prob
                    )
                    valid_mask = batch["sequence_mask"][:, :-1].bool()
                    rand_mask = rand_mask & valid_mask
                    replacements = predicted_classes[:, :-1].clamp(min=0)
                    forced_prev_tokens[:, 1:] = torch.where(
                        rand_mask,
                        replacements,
                        teacher_prev[:, :-1].clamp(min=0),
                    )
        lambda_prompt, lambda_target = compute_aggregate_weights(current_step, total_steps)
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
                cell_coords=batch["cell_coords"],
                forced_prev_tokens=forced_prev_tokens,
            )
            loss, metrics = compute_losses(
                outputs,
                batch,
                dataset,
                lambda_prompt=lambda_prompt,
                lambda_target=lambda_target,
            )
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
            logging.info("Epoch %s step %s: %s", epoch, step + 1, log_items)
            display_items = list(sorted(averages.items()))[:3]
            progress.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k, v in display_items})
            if writer is not None:
                for key, value in averages.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
            if writer is not None and (step + 1) % 100 == 0:
                writer.flush()
            logger = metrics_utils.MetricsLogger()
    remaining_metrics = logger.to_dict()
    if remaining_metrics and writer is not None:
        for key, value in remaining_metrics.items():
            writer.add_scalar(f"train/{key}", value, global_step)
    if writer is not None:
        writer.flush()
    return global_step


def evaluate(
    model: AutoregressiveTransformer,
    loader: DataLoader,
    dataset: GridDataset,
    device: torch.device,
    writer: Optional["SummaryWriter"] = None,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    logger = metrics_utils.MetricsLogger()
    lambda_prompt_eval = PROMPT_WEIGHT_END
    lambda_target_eval = max(1.0 - lambda_prompt_eval, 0.0)
    with torch.no_grad():
        progress = tqdm(loader, desc="Eval", leave=False, total=len(loader))
        for batch in progress:
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
                cell_coords=batch["cell_coords"],
            )
            loss, metrics = compute_losses(
                outputs,
                batch,
                dataset,
                lambda_prompt=lambda_prompt_eval,
                lambda_target=lambda_target_eval,
            )
            logger.update("loss/total", float(loss.item()))
            for key, value in metrics.items():
                logger.update(f"val_{key}", value)
            progress.set_postfix({"loss": f"{float(loss.item()):.3f}"})
    metrics = logger.to_dict()
    if writer is not None and global_step is not None:
        for key, value in metrics.items():
            writer.add_scalar(f"eval/{key}", value, global_step)
        writer.flush()
    return metrics



def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    out_dir = ensure_dir(args.out_dir)
    writer = create_summary_writer(out_dir, args)

    dataset = GridDataset(
        args.grid,
        args.descriptions,
        min_cells=args.min_cells,
        max_context=args.max_context,
        hf_token=args.hf_token,
    )

    train_indices, val_indices = stratified_split(dataset, args.val_split, args.seed)
    logging.info(
        "Dataset prepared with %d zones (%d train / %d val)",
        len(dataset),
        len(train_indices),
        len(val_indices),
    )
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
    logging.info("Using device: %s", device)

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
    try:
        for epoch in range(1, args.epochs + 1):
            logging.info("Starting epoch %d / %d", epoch, args.epochs)
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
                writer,
            )
            metrics = evaluate(model, val_loader, dataset, device, writer=writer, global_step=global_step)
            if metrics:
                log_items = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                logging.info("Epoch %s validation: %s", epoch, log_items)
            if writer is not None:
                writer.flush()
            if epoch % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, out_dir)
                logging.info("Checkpoint saved for epoch %d", epoch)
    finally:
        if writer is not None:
            writer.close()

    torch.save(model.state_dict(), out_dir / "model.pt")
    logging.info("Final model checkpoint saved to %s", out_dir / "model.pt")

    vocab = {
        "service_types": dataset.service_types,
        "zone_types": [dataset.id_to_zone_type[i] for i in sorted(dataset.id_to_zone_type)],
        "edge_tokens": EDGE_TOKENS,
    }
    save_json(vocab, out_dir / "vocab.json")
    logging.info("Vocabulary saved with %d service types and %d zone types", len(dataset.service_types), len(dataset.zone_type_to_id))

    norm_stats = {
        "living_mean": dataset.living_mean,
        "living_std": dataset.living_std,
        "service_capacity_mean": dataset.capacity_mean,
        "service_capacity_std": dataset.capacity_std,
        "cell_size_m": args.cell_size,
    }
    save_json(norm_stats, out_dir / "norm_stats.json")
    logging.info("Normalization statistics saved to %s", out_dir / "norm_stats.json")

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
    logging.info("Inference configuration saved to %s", out_dir / "inference_config.json")

    upload_artifacts_to_hf(out_dir, args)


if __name__ == "__main__":
    main()
