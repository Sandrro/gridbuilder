"""Inference script for the autoregressive grid Transformer with progress tracking, stage logging, and multi-zone parallel decoding."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data.dataset import _compute_directional_distances  # type: ignore
from models.transformer import AutoregressiveTransformer, ModelConfig
from utils.io import load_json

# optional tqdm progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# ---------------------- Logging setup ----------------------

def _parse_log_level(level_str: str) -> int:
    try:
        return getattr(logging, level_str.upper())
    except Exception:
        return logging.INFO

def _setup_logger(level: str, log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("infer")
    logger.setLevel(_parse_log_level(level))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(_parse_log_level(level))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(_parse_log_level(level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--config", required=True, help="Path to inference_config.json")

    # single-zone mode (backward compatible)
    parser.add_argument("--grid", help="Path to grid_cells.parquet for the target zone")
    parser.add_argument("--zone-id", help="Zone identifier within the grid Parquet")
    parser.add_argument("--zone-type", help="Zone type label for the single-zone mode")
    parser.add_argument("--budget-json", help="JSON file with living area and service budgets for the single zone")
    parser.add_argument("--out-parquet", help="Destination path for Parquet predictions (single zone)")
    parser.add_argument("--out-geojson", help="Optional GeoJSON output (single zone)")

    parser.add_argument("--zones-geojson", help="Optional GeoJSON for dynamic grid generation (not implemented)")
    parser.add_argument("--cell-size", type=float, default=15.0, help="Cell size in meters when building grids")

    # multi-zone parallel mode
    parser.add_argument("--batch-manifest", help="Path to JSON list of zone entries for parallel decoding. "
                                                 "Each entry: {"
                                                 "\"grid\": str, \"zone_id\": str, \"zone_type\": str, "
                                                 "\"budget_json\": str, \"out_parquet\": str, "
                                                 "optional \"out_geojson\": str}"
                                                 )
    parser.add_argument("--max-parallel", type=int, default=4, help="Max zones decoded in parallel (micro-batch size).")

    # sampling/decoding knobs
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument(
        "--budget-guidance-strength",
        type=float,
        default=1.0,
        help="Multiplier for budget-aware logit adjustments",
    )

    # progress/logging
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs")
    return parser.parse_args()


# ---------------------- Data utils ----------------------

def load_zone_from_grid(grid_path: str, zone_id: str) -> pd.DataFrame:
    df = pd.read_parquet(grid_path)
    zone_df = df[df["zone_id"] == zone_id].copy()
    if zone_df.empty:
        raise ValueError(f"Zone {zone_id} not found in {grid_path}")
    zone_df.sort_values(["ring_index", "ring_order"], inplace=True)
    zone_df.reset_index(drop=True, inplace=True)
    return zone_df


# ---------------------- Sampling ----------------------

def nucleus_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    if cutoff.any():
        first_idx = torch.nonzero(cutoff, as_tuple=False)[0, 0]
        sorted_probs = sorted_probs[: first_idx + 1]
        sorted_indices = sorted_indices[: first_idx + 1]
        probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(probs, num_samples=1)
        return int(sorted_indices[choice])
    choice = torch.multinomial(probs, num_samples=1)
    return int(choice)


# ---------------------- Prompts ----------------------

def prepare_prompts(
    vocab: Dict[str, List[str]],
    norm_stats: Dict[str, object],
    zone_type: str,
    budgets: Dict[str, object],
    num_cells: int,
) -> Dict[str, Any]:
    zone_types = vocab["zone_types"]
    if zone_type not in zone_types:
        raise ValueError(f"Unknown zone type {zone_type}; available: {zone_types}")
    zone_type_id = zone_types.index(zone_type)

    living_budget = float(budgets.get("living_area", 0.0) or 0.0)
    if living_budget > 0 and num_cells > 0:
        density = np.log1p(living_budget / num_cells)
        living_mean = float(norm_stats.get("living_mean", 0.0))
        living_std = max(float(norm_stats.get("living_std", 1.0)), 1e-6)
        living_prompt = (density - living_mean) / living_std
        living_mask = 1.0
    else:
        living_prompt = 0.0
        living_mask = 0.0

    service_types: List[str] = vocab.get("service_types", [])
    service_prompt = np.zeros(len(service_types), dtype=np.float32)
    service_prompt_mask = np.zeros(len(service_types), dtype=np.float32)
    service_budgets: Dict[str, float] = budgets.get("services", {}) if isinstance(budgets.get("services"), dict) else {}
    means: Dict[str, float] = norm_stats.get("service_capacity_mean", {})  # type: ignore[assignment]
    stds: Dict[str, float] = norm_stats.get("service_capacity_std", {})  # type: ignore[assignment]
    for idx, service in enumerate(service_types):
        value = float(service_budgets.get(service, 0.0) or 0.0)
        if value <= 0:
            continue
        density = np.log1p(value / max(num_cells, 1))
        mean = float(means.get(service, 0.0))
        std = max(float(stds.get(service, 1.0)), 1e-6)
        service_prompt[idx] = (density - mean) / std
        service_prompt_mask[idx] = 1.0
    return {
        "zone_type_id": torch.tensor([zone_type_id], dtype=torch.long),
        "living_prompt": torch.tensor([[living_prompt]], dtype=torch.float32),
        "living_prompt_mask": torch.tensor([[living_mask]], dtype=torch.float32),
        "service_prompt": torch.from_numpy(service_prompt).unsqueeze(0),
        "service_prompt_mask": torch.from_numpy(service_prompt_mask).unsqueeze(0),
        "living_budget_total": living_budget,
        "service_budgets": service_budgets,
    }


# ---------------------- Budget tracker ----------------------

class BudgetTracker:
    """Tracks remaining budgets and produces logit adjustments."""

    def __init__(
        self,
        living_total: float,
        service_totals: Dict[str, float],
        service_types: List[str],
        total_steps: int,
        guidance_strength: float,
    ) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.guidance_strength = float(max(guidance_strength, 0.0))

        self.living_total = float(living_total)
        self.living_remaining = float(living_total)
        self.living_target_density = (
            self.living_total / self.total_steps if self.total_steps > 0 else 0.0
        )

        self.service_types = service_types
        self.service_remaining = {
            service: float(service_totals.get(service, 0.0) or 0.0) for service in service_types
        }
        self.service_target_density = {
            service: (
                self.service_remaining[service] / self.total_steps if self.total_steps > 0 else 0.0
            )
            for service in service_types
        }
        total_service = float(sum(self.service_remaining.values()))
        self.service_total_target_density = (
            total_service / self.total_steps if self.total_steps > 0 else 0.0
        )

    @staticmethod
    def _pressure_ratio(remaining: float, target_density: float, steps_left: int) -> float:
        if remaining <= 0:
            return 0.0
        if steps_left <= 0:
            return float("inf")
        demand_density = remaining / steps_left
        if target_density <= 0:
            return float("inf")
        return demand_density / target_density

    def _progress_nudge(self, steps_left: int) -> float:
        if self.total_steps <= 0:
            return 0.0
        progress = 1.0 - float(steps_left) / float(self.total_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.guidance_strength * progress

    def guidance_for_class(self, class_logits: torch.Tensor, steps_left: int) -> torch.Tensor:
        adjusted = class_logits.clone()
        steps = max(int(steps_left), 1)
        base_push = self._progress_nudge(steps)

        if self.living_remaining > 0:
            adjusted[1] += base_push
            adjusted[3] += base_push
            ratio = self._pressure_ratio(self.living_remaining, self.living_target_density, steps)
            if ratio > 1.0:
                delta = self.guidance_strength * math.log1p(ratio - 1.0)
                adjusted[1] += delta
                adjusted[3] += delta
                adjusted[0] -= delta

        service_remaining_total = self.service_remaining_sum()
        if service_remaining_total > 0:
            adjusted[2] += base_push
            adjusted[3] += base_push
            ratio = self._pressure_ratio(
                service_remaining_total, self.service_total_target_density, steps
            )
            if ratio > 1.0:
                delta = self.guidance_strength * math.log1p(ratio - 1.0)
                adjusted[2] += delta
                adjusted[3] += delta
                adjusted[0] -= delta

        if (self.living_remaining > 0 or service_remaining_total > 0) and steps <= 1:
            adjusted[0] -= max(self.guidance_strength, 1.0) * 1e6

        return adjusted

    def guidance_for_service(self, service_logits: torch.Tensor, steps_left: int) -> torch.Tensor:
        if not self.service_types:
            return service_logits

        adjusted = service_logits.clone()
        steps = max(int(steps_left), 1)
        base_push = self._progress_nudge(steps)
        num_types = len(self.service_types)
        total_remaining = 0.0

        for idx, service in enumerate(self.service_types):
            remaining = self.service_remaining.get(service, 0.0)
            if remaining <= 0:
                continue
            total_remaining += remaining
            adjusted[idx] += base_push
            ratio = self._pressure_ratio(
                remaining, self.service_target_density.get(service, 0.0), steps
            )
            if ratio > 1.0:
                delta = self.guidance_strength * math.log1p(ratio - 1.0)
                adjusted[idx] += delta
                if num_types > 1:
                    penalty = delta / (num_types - 1)
                    for j in range(num_types):
                        if j != idx:
                            adjusted[j] -= penalty

        if total_remaining > 0 and steps <= 1:
            max_idx = 0
            max_remaining = -1.0
            for idx, service in enumerate(self.service_types):
                remaining = self.service_remaining.get(service, 0.0)
                if remaining > max_remaining:
                    max_remaining = remaining
                    max_idx = idx
            large = max(self.guidance_strength, 1.0) * 1e6
            for j in range(num_types):
                if j == max_idx:
                    adjusted[j] += large
                else:
                    adjusted[j] -= large

        return adjusted

    def update(self, class_id: int, living_area: float, service_type: Optional[str], service_capacity: float) -> None:
        if class_id in (1, 3):
            self.living_remaining = max(self.living_remaining - living_area, 0.0)
        if class_id in (2, 3) and service_type is not None:
            self.service_remaining[service_type] = max(self.service_remaining.get(service_type, 0.0) - service_capacity, 0.0)

    def service_remaining_sum(self) -> float:
        return float(sum(self.service_remaining.values()))


# ---------------------- Progress helpers ----------------------

def _should_use_tqdm(no_progress_flag: bool) -> bool:
    if no_progress_flag:
        return False
    if tqdm is None:
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


# ---------------------- Core: single-zone decode ----------------------

def _decode_single_zone(
    logger: logging.Logger,
    model: AutoregressiveTransformer,
    device: torch.device,
    vocab: Dict[str, Any],
    norm_stats: Dict[str, Any],
    grid_path: str,
    zone_id: str,
    zone_type: str,
    budget_json: str,
    temperature: float,
    top_p: float,
    beam: int,
    no_progress: bool,
    out_parquet: str,
    out_geojson: Optional[str] = None,
    budget_guidance_strength: float = 1.0,
) -> None:
    logger.info("Reading grid parquet for zone_id=%s from %s", zone_id, grid_path)
    zone_df = load_zone_from_grid(grid_path, zone_id)
    rows = zone_df["row"].to_numpy()
    cols = zone_df["col"].to_numpy()
    ring_index = zone_df["ring_index"].to_numpy()
    logger.debug("Zone df shape: %s, num_cells=%d", zone_df.shape, len(zone_df))

    logger.info("Computing directional distances...")
    distances = _compute_directional_distances(rows, cols, ring_index)
    num_cells = len(zone_df)
    logger.debug("Distances shape: %s", distances.shape)

    budgets = load_json(budget_json)
    prompts = prepare_prompts(vocab, norm_stats, zone_type, budgets, num_cells)
    logger.debug("Prompts: living_total=%.3f, service_keys=%s",
                 prompts["living_budget_total"], list(prompts["service_budgets"].keys()))

    sequence_mask = torch.zeros(1, num_cells, dtype=torch.float32, device=device)
    edge_distances = torch.from_numpy(distances).unsqueeze(0).to(device)
    generated_classes = torch.zeros(1, num_cells, dtype=torch.long, device=device)

    tracker = BudgetTracker(
        prompts["living_budget_total"],
        prompts["service_budgets"],
        vocab.get("service_types", []),
        total_steps=num_cells,
        guidance_strength=budget_guidance_strength,
    )

    outputs: List[Dict[str, Any]] = []
    remaining_steps = num_cells

    logger.info("Starting decoding (single) | num_cells=%d, temperature=%.3f, top_p=%.3f, beam=%d",
                num_cells, temperature, top_p, beam)
    use_bar = _should_use_tqdm(no_progress=no_progress)
    bar = tqdm(total=num_cells, unit="cell", desc="Decoding cells", leave=False) if use_bar else None
    text_update_every = max(1, num_cells // 20)

    for idx in range(num_cells):
        sequence_mask[0, idx] = 1.0
        with torch.no_grad():
            model_outputs = model.decode_step(
                generated_classes,
                zone_type_ids=prompts["zone_type_id"].to(device),
                living_prompt=prompts["living_prompt"].to(device),
                living_prompt_mask=prompts["living_prompt_mask"].to(device),
                service_prompt=prompts["service_prompt"].to(device),
                service_prompt_mask=prompts["service_prompt_mask"].to(device),
                edge_distances=edge_distances,
                sequence_mask=sequence_mask,
            )

        cell_logits = model_outputs["cell_class_logits"][0, idx]
        guided_logits = tracker.guidance_for_class(cell_logits, remaining_steps)
        class_id = nucleus_sample(guided_logits, top_p, temperature)
        generated_classes[0, idx] = class_id

        living_prob = torch.sigmoid(model_outputs["is_living_logits"][0, idx]).item()
        storeys = model_outputs["storeys"][0, idx].item()
        living_area = F.softplus(model_outputs["living_area"][0, idx]).item()
        service_capacity_vector = F.softplus(model_outputs["service_capacity"][0, idx])
        service_capacity = 0.0
        service_type = None

        if class_id in (2, 3) and vocab.get("service_types"):
            service_logits = model_outputs["service_type_logits"][0, idx]
            guided_service_logits = tracker.guidance_for_service(service_logits, remaining_steps)
            service_type_id = nucleus_sample(guided_service_logits, top_p, temperature)
            service_type = vocab["service_types"][service_type_id]
            service_capacity = float(service_capacity_vector[service_type_id].item())
        else:
            service_capacity = 0.0

        if class_id == 0:
            living_area = 0.0
            storeys = 0.0
            living_prob = 0.0
            service_capacity = 0.0
            service_type = None
        elif class_id == 1:
            service_capacity = 0.0
            service_type = None
        elif class_id == 2:
            living_area = 0.0
            storeys = 0.0
            living_prob = 0.0

        tracker.update(class_id, living_area, service_type, service_capacity)
        remaining_steps -= 1

        if bar is not None:
            bar.update(1)
            if (idx + 1) % 10 == 0 or idx == num_cells - 1:
                bar.set_description(
                    f"Decoding cells | living_rem={tracker.living_remaining:.1f}, "
                    f"services_rem={tracker.service_remaining_sum():.1f}"
                )
        else:
            if (idx + 1) % text_update_every == 0 or idx == num_cells - 1:
                logging.getLogger("infer").info(
                    "[progress] %d/%d | living_rem=%.1f | services_rem=%.1f",
                    idx + 1, num_cells, tracker.living_remaining, tracker.service_remaining_sum()
                )

        outputs.append({
            "zone_id": zone_id,
            "cell_id": zone_df.loc[idx, "cell_id"],
            "row": int(zone_df.loc[idx, "row"]),
            "col": int(zone_df.loc[idx, "col"]),
            "ring_index": int(zone_df.loc[idx, "ring_index"]),
            "ring_order": int(zone_df.loc[idx, "ring_order"]),
            "cell_class": int(class_id),
            "is_living_prob": float(living_prob),
            "storeys": float(max(storeys, 0.0)),
            "living_area_cell": float(max(living_area, 0.0)),
            "service_type": service_type,
            "service_capacity_cell": float(max(service_capacity, 0.0)),
        })

    if bar is not None:
        bar.close()
    logging.getLogger("infer").info("Decoding finished.")

    pred_df = pd.DataFrame(outputs)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_parquet, index=False)
    logging.getLogger("infer").info("Saved predictions parquet: %s (rows=%d)", out_parquet, len(pred_df))

    if out_geojson:
        logging.getLogger("infer").info("Assembling GeoJSON to: %s", out_geojson)
        features = []
        geometries = zone_df.get("geometry")
        if geometries is not None:
            try:
                from shapely import wkb
                from shapely.geometry import mapping
            except Exception:
                geometries = None
        for row_idx, row in pred_df.iterrows():
            properties = row.to_dict()
            geometry = None
            if geometries is not None:
                try:
                    geom = wkb.loads(zone_df.loc[row_idx, "geometry"], hex=False)
                    geometry = mapping(geom)
                except Exception:
                    geometry = None
            features.append({"type": "Feature", "geometry": geometry, "properties": properties})
        geojson = {"type": "FeatureCollection", "features": features}
        Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
        Path(out_geojson).write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.getLogger("infer").info("Saved GeoJSON: %s (features=%d)", out_geojson, len(features))


# ---------------------- Core: multi-zone parallel decode ----------------------

@dataclass
class _ZonePack:
    zone_id: str
    zone_type: str
    zone_df: pd.DataFrame
    distances: np.ndarray
    prompts: Dict[str, Any]
    tracker: BudgetTracker
    T: int  # number of cells
    out_parquet: str
    out_geojson: Optional[str]

def _prepare_zone_pack(
    vocab: Dict[str, Any],
    norm_stats: Dict[str, Any],
    entry: Dict[str, Any],
    logger: logging.Logger,
    budget_guidance_strength: float,
) -> _ZonePack:
    grid = entry["grid"]
    zone_id = str(entry["zone_id"])
    zone_type = str(entry["zone_type"])
    budget_json = entry["budget_json"]
    out_parquet = entry["out_parquet"]
    out_geojson = entry.get("out_geojson")

    zone_df = load_zone_from_grid(grid, zone_id)
    rows = zone_df["row"].to_numpy()
    cols = zone_df["col"].to_numpy()
    ring_index = zone_df["ring_index"].to_numpy()
    distances = _compute_directional_distances(rows, cols, ring_index)
    T = len(zone_df)

    budgets = load_json(budget_json)
    prompts = prepare_prompts(vocab, norm_stats, zone_type, budgets, T)
    tracker = BudgetTracker(
        prompts["living_budget_total"],
        prompts["service_budgets"],
        vocab.get("service_types", []),
        total_steps=T,
        guidance_strength=budget_guidance_strength,
    )

    logger.debug("Prepared zone %s | T=%d | zone_type=%s", zone_id, T, zone_type)
    return _ZonePack(zone_id, zone_type, zone_df, distances, prompts, tracker, T, out_parquet, out_geojson)


def _decode_multi_zone_batch(
    logger: logging.Logger,
    model: AutoregressiveTransformer,
    device: torch.device,
    vocab: Dict[str, Any],
    packs: List[_ZonePack],
    temperature: float,
    top_p: float,
    beam: int,
    no_progress: bool,
) -> None:
    """Decode several zones in parallel (micro-batch)."""
    B = len(packs)
    T_max = max(p.T for p in packs)

    # stack / pad tensors
    dir_dim = packs[0].distances.shape[1] if packs[0].distances.ndim == 2 else packs[0].distances.shape[-1]
    edge_distances = np.zeros((B, T_max, dir_dim), dtype=np.float32)
    sequence_mask = torch.zeros(B, T_max, dtype=torch.float32, device=device)
    generated_classes = torch.zeros(B, T_max, dtype=torch.long, device=device)

    lengths = np.array([p.T for p in packs], dtype=np.int32)

    for i, p in enumerate(packs):
        edge_distances[i, :p.T, :] = p.distances
        # masks will be filled step-wise

    edge_distances = torch.from_numpy(edge_distances).to(device)

    # Prepare constant prompt tensors per sample
    zone_type_ids = torch.cat([p.prompts["zone_type_id"] for p in packs], dim=0).to(device)  # [B]
    living_prompt = torch.cat([p.prompts["living_prompt"] for p in packs], dim=0).to(device)  # [B,1]
    living_prompt_mask = torch.cat([p.prompts["living_prompt_mask"] for p in packs], dim=0).to(device)  # [B,1]
    service_prompt = torch.cat([p.prompts["service_prompt"] for p in packs], dim=0).to(device)  # [B,S]
    service_prompt_mask = torch.cat([p.prompts["service_prompt_mask"] for p in packs], dim=0).to(device)  # [B,S]

    # progress
    total_steps = int(lengths.sum())
    use_bar = _should_use_tqdm(no_progress)
    bar = tqdm(total=total_steps, unit="cell", desc="Decoding cells (batch)", leave=False) if use_bar else None
    text_update_every = max(1, total_steps // 20)
    progressed = 0

    outputs_per_zone: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
    remaining_steps = lengths.copy()

    logger.info("Starting decoding (batch) | zones=%d | total_cells=%d | T_max=%d | temp=%.3f top_p=%.3f",
                B, total_steps, T_max, temperature, top_p)

    for t in range(T_max):
        # enable current timestep for active sequences
        for i in range(B):
            if t < lengths[i]:
                sequence_mask[i, t] = 1.0

        with torch.no_grad():
            mo = model.decode_step(
                generated_classes,
                zone_type_ids=zone_type_ids,
                living_prompt=living_prompt,
                living_prompt_mask=living_prompt_mask,
                service_prompt=service_prompt,
                service_prompt_mask=service_prompt_mask,
                edge_distances=edge_distances,
                sequence_mask=sequence_mask,
            )

        # for each active sample at time t: sample and update trackers
        for i, p in enumerate(packs):
            if t >= lengths[i]:
                continue  # padding time step

            cell_logits = mo["cell_class_logits"][i, t]
            guided_logits = p.tracker.guidance_for_class(cell_logits, int(remaining_steps[i]))
            class_id = nucleus_sample(guided_logits, top_p, temperature)
            generated_classes[i, t] = class_id

            living_prob = torch.sigmoid(mo["is_living_logits"][i, t]).item()
            storeys = mo["storeys"][i, t].item()
            living_area = F.softplus(mo["living_area"][i, t]).item()
            service_capacity_vec = F.softplus(mo["service_capacity"][i, t])
            service_capacity = 0.0
            service_type = None

            if class_id in (2, 3) and vocab.get("service_types"):
                service_logits = mo["service_type_logits"][i, t]
                guided_service_logits = p.tracker.guidance_for_service(service_logits, int(remaining_steps[i]))
                service_type_id = nucleus_sample(guided_service_logits, top_p, temperature)
                service_type = vocab["service_types"][service_type_id]
                service_capacity = float(service_capacity_vec[service_type_id].item())
            else:
                service_capacity = 0.0

            if class_id == 0:
                living_area = 0.0
                storeys = 0.0
                living_prob = 0.0
                service_capacity = 0.0
                service_type = None
            elif class_id == 1:
                service_capacity = 0.0
                service_type = None
            elif class_id == 2:
                living_area = 0.0
                storeys = 0.0
                living_prob = 0.0

            p.tracker.update(class_id, living_area, service_type, service_capacity)
            remaining_steps[i] -= 1

            outputs_per_zone[i].append({
                "zone_id": p.zone_id,
                "cell_id": p.zone_df.loc[t, "cell_id"],
                "row": int(p.zone_df.loc[t, "row"]),
                "col": int(p.zone_df.loc[t, "col"]),
                "ring_index": int(p.zone_df.loc[t, "ring_index"]),
                "ring_order": int(p.zone_df.loc[t, "ring_order"]),
                "cell_class": int(class_id),
                "is_living_prob": float(living_prob),
                "storeys": float(max(storeys, 0.0)),
                "living_area_cell": float(max(living_area, 0.0)),
                "service_type": service_type,
                "service_capacity_cell": float(max(service_capacity, 0.0)),
            })

            progressed += 1
            if bar is not None:
                bar.update(1)
            else:
                if progressed % text_update_every == 0 or progressed == total_steps:
                    logger.info(
                        "[progress-batch] %d/%d | zone=%s | living_rem=%.1f | services_rem=%.1f",
                        progressed, total_steps, p.zone_id, p.tracker.living_remaining, p.tracker.service_remaining_sum()
                    )

    if bar is not None:
        bar.close()

    # write outputs
    for i, p in enumerate(packs):
        pred_df = pd.DataFrame(outputs_per_zone[i])
        Path(p.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(p.out_parquet, index=False)
        logger.info("Saved predictions parquet for zone %s: %s (rows=%d)", p.zone_id, p.out_parquet, len(pred_df))

        if p.out_geojson:
            logger.info("Assembling GeoJSON for zone %s -> %s", p.zone_id, p.out_geojson)
            features = []
            geometries = p.zone_df.get("geometry")
            if geometries is not None:
                try:
                    from shapely import wkb
                    from shapely.geometry import mapping
                except Exception:
                    geometries = None
            for row_idx, row in pred_df.iterrows():
                properties = row.to_dict()
                geometry = None
                if geometries is not None:
                    try:
                        geom = wkb.loads(p.zone_df.loc[row_idx, "geometry"], hex=False)
                        geometry = mapping(geom)
                    except Exception:
                        geometry = None
                features.append({"type": "Feature", "geometry": geometry, "properties": properties})
            fc = {"type": "FeatureCollection", "features": features}
            Path(p.out_geojson).parent.mkdir(parents=True, exist_ok=True)
            Path(p.out_geojson).write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Saved GeoJSON for zone %s: features=%d", p.zone_id, len(features))


# ---------------------- Main ----------------------

def main() -> None:
    args = parse_args()
    logger = _setup_logger(args.log_level, args.log_file)

    # Load config/vocab/stats
    logger.info("Loading config/vocab/norm_stats...")
    vocab = load_json(args.vocab)
    norm_stats = load_json(Path(args.config).with_name("norm_stats.json"))
    config_dict = load_json(args.config)
    config = ModelConfig(**config_dict["model"])
    logger.debug("Config: %s", config_dict["model"])
    logger.debug("Vocab sizes: zone_types=%d, service_types=%d",
                 len(vocab.get("zone_types", [])), len(vocab.get("service_types", [])))

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Building model on device: %s", device)
    model = AutoregressiveTransformer(
        config,
        service_vocab=len(vocab.get("service_types", [])),
        zone_vocab=len(vocab.get("zone_types", [])),
    ).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint from: %s", args.model)
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    logger.info("[ckpt] loaded. missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logger.debug("[ckpt] missing (first 10): %s", missing[:10])
    if unexpected:
        logger.debug("[ckpt] unexpected (first 10): %s", unexpected[:10])
    model.eval()

    # Choose mode
    if args.batch_manifest:
        # MULTI-ZONE PARALLEL MODE
        with open(args.batch_manifest, "r", encoding="utf-8") as f:
            manifest: List[Dict[str, Any]] = json.load(f)
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("--batch-manifest must be a non-empty JSON list of entries")

        # process in chunks of max_parallel
        max_parallel = max(1, int(args.max_parallel))
        logger.info("Parallel mode: max_parallel=%d | entries=%d", max_parallel, len(manifest))

        # prepare packs lazily per chunk to limit CPU/RAM
        idx = 0
        while idx < len(manifest):
            chunk_entries = manifest[idx: idx + max_parallel]
            packs: List[_ZonePack] = []
            for e in chunk_entries:
                packs.append(
                    _prepare_zone_pack(
                        vocab,
                        norm_stats,
                        e,
                        logger,
                        budget_guidance_strength=args.budget_guidance_strength,
                    )
                )
            _decode_multi_zone_batch(
                logger=logger,
                model=model,
                device=device,
                vocab=vocab,
                packs=packs,
                temperature=args.temperature,
                top_p=args.top_p,
                beam=args.beam,
                no_progress=args.no_progress,
            )
            idx += max_parallel

        logger.info("All batches finished.")
    else:
        # SINGLE-ZONE MODE (backward compatible)
        if not args.grid or not args.zone_id or not args.zone_type or not args.budget_json or not args.out_parquet:
            logger.error("Single-zone mode requires --grid, --zone-id, --zone-type, --budget-json, --out-parquet")
            raise ValueError("Missing required arguments for single-zone mode.")
        if args.zones_geojson:
            logger.error("GeoJSON-based grid generation is not implemented in this reference script")
            raise NotImplementedError("GeoJSON-based grid generation is not implemented in this reference script")

        _decode_single_zone(
            logger=logger,
            model=model,
            device=device,
            vocab=vocab,
            norm_stats=norm_stats,
            grid_path=args.grid,
            zone_id=args.zone_id,
            zone_type=args.zone_type,
            budget_json=args.budget_json,
            temperature=args.temperature,
            top_p=args.top_p,
            beam=args.beam,
            no_progress=args.no_progress,
            out_parquet=args.out_parquet,
            out_geojson=args.out_geojson,
            budget_guidance_strength=args.budget_guidance_strength,
        )

    # Done


if __name__ == "__main__":
    main()
