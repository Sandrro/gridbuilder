"""Inference script for the autoregressive grid Transformer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data.dataset import _compute_directional_distances  # type: ignore
from models.transformer import AutoregressiveTransformer, ModelConfig
from utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--config", required=True, help="Path to inference_config.json")
    parser.add_argument("--grid", help="Path to grid_cells.parquet for the target zone")
    parser.add_argument("--zone-id", help="Zone identifier within the grid Parquet")
    parser.add_argument("--zones-geojson", help="Optional GeoJSON for dynamic grid generation")
    parser.add_argument("--cell-size", type=float, default=15.0, help="Cell size in meters when building grids")
    parser.add_argument("--zone-type", required=True, help="Zone type label")
    parser.add_argument("--budget-json", required=True, help="JSON file with living area and service budgets")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--out-parquet", required=True, help="Destination path for Parquet predictions")
    parser.add_argument("--out-geojson", help="Optional GeoJSON output")
    return parser.parse_args()


def load_zone_from_grid(grid_path: str, zone_id: str) -> pd.DataFrame:
    df = pd.read_parquet(grid_path)
    zone_df = df[df["zone_id"] == zone_id].copy()
    if zone_df.empty:
        raise ValueError(f"Zone {zone_id} not found in {grid_path}")
    zone_df.sort_values(["ring_index", "ring_order"], inplace=True)
    zone_df.reset_index(drop=True, inplace=True)
    return zone_df


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


def prepare_prompts(
    vocab: Dict[str, List[str]],
    norm_stats: Dict[str, object],
    zone_type: str,
    budgets: Dict[str, object],
    num_cells: int,
) -> Dict[str, torch.Tensor]:
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


class BudgetTracker:
    """Tracks remaining budgets and produces gentle logit nudges."""

    def __init__(self, living_total: float, service_totals: Dict[str, float], service_types: List[str]) -> None:
        self.living_remaining = living_total
        self.service_remaining = {service: float(service_totals.get(service, 0.0) or 0.0) for service in service_types}
        self.service_types = service_types

    def guidance_for_class(self, class_logits: torch.Tensor, steps_left: int) -> torch.Tensor:
        adjustment = torch.zeros_like(class_logits)
        urgency = 1.0 - steps_left / max(steps_left + 1, 1)
        if self.living_remaining > 0:
            adjustment[1] += 0.2 * urgency
            adjustment[3] += 0.2 * urgency
        if any(value > 0 for value in self.service_remaining.values()):
            adjustment[2] += 0.2 * urgency
            adjustment[3] += 0.2 * urgency
        return class_logits + adjustment

    def guidance_for_service(self, service_logits: torch.Tensor, steps_left: int) -> torch.Tensor:
        adjustment = torch.zeros_like(service_logits)
        urgency = 1.0 - steps_left / max(steps_left + 1, 1)
        for idx, service in enumerate(self.service_types):
            remaining = self.service_remaining.get(service, 0.0)
            if remaining > 0:
                adjustment[idx] += 0.2 * urgency
        return service_logits + adjustment

    def update(self, class_id: int, living_area: float, service_type: Optional[str], service_capacity: float) -> None:
        if class_id in (1, 3):
            self.living_remaining = max(self.living_remaining - living_area, 0.0)
        if class_id in (2, 3) and service_type is not None:
            self.service_remaining[service_type] = max(self.service_remaining.get(service_type, 0.0) - service_capacity, 0.0)


def main() -> None:
    args = parse_args()

    if args.grid and not args.zone_id:
        raise ValueError("--zone-id is required when --grid is provided")
    if not args.grid and not args.zones_geojson:
        raise ValueError("Either --grid or --zones-geojson must be supplied")
    if args.zones_geojson:
        raise NotImplementedError("GeoJSON-based grid generation is not implemented in this reference script")

    vocab = load_json(args.vocab)
    norm_stats = load_json(Path(args.config).with_name("norm_stats.json"))
    config_dict = load_json(args.config)
    config = ModelConfig(**config_dict["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoregressiveTransformer(
        config,
        service_vocab=len(vocab.get("service_types", [])),
        zone_vocab=len(vocab.get("zone_types", [])),
    ).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.grid:
        zone_df = load_zone_from_grid(args.grid, args.zone_id)
    else:
        raise NotImplementedError("GeoJSON-based grid generation is not implemented in this reference script")

    rows = zone_df["row"].to_numpy()
    cols = zone_df["col"].to_numpy()
    ring_index = zone_df["ring_index"].to_numpy()
    distances = _compute_directional_distances(rows, cols, ring_index)
    num_cells = len(zone_df)

    budgets = load_json(args.budget_json)
    prompts = prepare_prompts(vocab, norm_stats, args.zone_type, budgets, num_cells)

    sequence_mask = torch.zeros(1, num_cells, dtype=torch.float32, device=device)
    edge_distances = torch.from_numpy(distances).unsqueeze(0).to(device)
    generated_classes = torch.zeros(1, num_cells, dtype=torch.long, device=device)

    tracker = BudgetTracker(prompts["living_budget_total"], prompts["service_budgets"], vocab.get("service_types", []))

    outputs: List[Dict[str, object]] = []
    remaining_steps = num_cells
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
        class_id = nucleus_sample(guided_logits, args.top_p, args.temperature)
        generated_classes[0, idx] = class_id

        living_prob = torch.sigmoid(model_outputs["is_living_logits"][0, idx]).item()
        storeys = model_outputs["storeys"][0, idx].item()
        living_area = F.softplus(model_outputs["living_area"][0, idx]).item()
        service_capacity = F.softplus(model_outputs["service_capacity"][0, idx]).item()

        service_type = None
        if class_id in (2, 3) and vocab.get("service_types"):
            service_logits = model_outputs["service_type_logits"][0, idx]
            guided_service_logits = tracker.guidance_for_service(service_logits, remaining_steps)
            service_type_id = nucleus_sample(guided_service_logits, args.top_p, args.temperature)
            service_type = vocab["service_types"][service_type_id]
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

        outputs.append(
            {
                "zone_id": args.zone_id,
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
            }
        )

    pred_df = pd.DataFrame(outputs)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(args.out_parquet, index=False)

    if args.out_geojson:
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
        Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_geojson).write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")

    living_total = pred_df["living_area_cell"].sum()
    service_totals = pred_df.groupby("service_type")["service_capacity_cell"].sum().to_dict()
    print("Predicted living area:", living_total)
    print("Predicted service totals:", service_totals)


if __name__ == "__main__":
    main()
