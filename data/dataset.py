"""Dataset utilities for the autoregressive grid Transformer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1), # NW
)

EDGE_TOKENS = [
    "EDGE_N",
    "EDGE_NE",
    "EDGE_E",
    "EDGE_SE",
    "EDGE_S",
    "EDGE_SW",
    "EDGE_W",
    "EDGE_NW",
]


@dataclass
class ZoneExample:
    """Container for a single zone example."""

    zone_id: str
    zone_type_id: int
    prompt: Dict[str, torch.Tensor]
    cell_class: torch.LongTensor
    is_living: torch.FloatTensor
    is_living_mask: torch.FloatTensor
    storeys: torch.FloatTensor
    storeys_mask: torch.FloatTensor
    living_area: torch.FloatTensor
    living_area_mask: torch.FloatTensor
    service_type: torch.LongTensor
    service_type_mask: torch.FloatTensor
    service_capacity: torch.FloatTensor
    service_capacity_mask: torch.FloatTensor
    edge_distances: torch.FloatTensor


def _compute_directional_distances(rows: np.ndarray, cols: np.ndarray, ring_index: np.ndarray) -> np.ndarray:
    """Compute distances to the boundary along the eight cardinal/diagonal directions."""
    coords = list(zip(rows.tolist(), cols.tolist()))
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    boundary = {coord for coord, r in zip(coords, ring_index.tolist()) if r == 0}
    distances = np.zeros((len(coords), len(DIRECTIONS)), dtype=np.float32)
    for idx, (r, c) in enumerate(coords):
        for dir_id, (dr, dc) in enumerate(DIRECTIONS):
            steps = 0
            rr, cc = r, c
            while True:
                rr += dr
                cc += dc
                coord = (rr, cc)
                if coord not in coord_to_idx:
                    break
                steps += 1
                if coord in boundary:
                    break
            distances[idx, dir_id] = steps
    return distances


def _parse_service_json(raw: str | float | None) -> Dict[str, float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if v is not None}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    result: Dict[str, float] = {}
    for key, value in data.items():
        if value is None:
            continue
        result[str(key)] = float(value)
    return result


class GridDataset(Dataset):
    """PyTorch dataset that yields :class:`ZoneExample` objects."""

    def __init__(
        self,
        grid_path: str,
        descriptions_path: str,
        *,
        min_cells: int = 16,
        max_context: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_df = pd.read_parquet(grid_path)
        self.desc_df = pd.read_parquet(descriptions_path)
        self.min_cells = min_cells
        self.max_context = max_context

        self.zone_groups = self.grid_df.groupby("zone_id")
        self.zone_meta = {row.zone_id: row for row in self.desc_df.itertuples(index=False)}

        self.zone_ids: List[str] = []
        self.service_types: List[str] = []
        self.zone_type_to_id: Dict[str, int] = {}
        self.id_to_zone_type: Dict[int, str] = {}

        self._prepare_vocabularies()
        self._collect_normalization_stats()

    def _prepare_vocabularies(self) -> None:
        service_types: set[str] = set()
        for name in self.grid_df["service_type_name"].dropna().unique():
            service_types.add(str(name))
        self.service_types = sorted(service_types)
        zone_types = sorted({str(row.zone_type) for row in self.desc_df.itertuples(index=False)})
        self.zone_type_to_id = {z: i for i, z in enumerate(zone_types)}
        self.id_to_zone_type = {i: z for z, i in self.zone_type_to_id.items()}
        for zone_id, group in self.zone_groups:
            if len(group) < self.min_cells:
                continue
            self.zone_ids.append(zone_id)

    def _collect_normalization_stats(self) -> None:
        living = []
        capacities: Dict[str, List[float]] = {service: [] for service in self.service_types}
        for zone_id in self.zone_ids:
            group = self.zone_groups.get_group(zone_id)
            meta = self.zone_meta.get(zone_id)
            if meta is None:
                continue
            num_cells = len(group)
            total_living = getattr(meta, "total_living_area", float("nan"))
            if total_living and not np.isnan(total_living) and total_living > 0:
                living.append(np.log1p(float(total_living) / num_cells))
            service_cap = _parse_service_json(getattr(meta, "service_capacities", None))
            for service, value in service_cap.items():
                if service not in capacities:
                    continue
                if value > 0:
                    capacities[service].append(np.log1p(float(value) / num_cells))
        self.living_mean = float(np.mean(living)) if living else 0.0
        self.living_std = float(np.std(living)) if living else 1.0
        self.capacity_mean: Dict[str, float] = {}
        self.capacity_std: Dict[str, float] = {}
        for service in self.service_types:
            values = capacities.get(service, [])
            if values:
                self.capacity_mean[service] = float(np.mean(values))
                self.capacity_std[service] = float(np.std(values))
            else:
                self.capacity_mean[service] = 0.0
                self.capacity_std[service] = 1.0

    def __len__(self) -> int:
        return len(self.zone_ids)

    def __getitem__(self, index: int) -> ZoneExample:
        zone_id = self.zone_ids[index]
        group = self.zone_groups.get_group(zone_id).sort_values([
            "ring_index",
            "ring_order",
        ])
        if self.max_context is not None:
            group = group.head(self.max_context)
        rows = group["row"].to_numpy()
        cols = group["col"].to_numpy()
        ring_index = group["ring_index"].to_numpy()
        distances = _compute_directional_distances(rows, cols, ring_index)

        zone_type = str(self.zone_meta[zone_id].zone_type)
        zone_type_id = self.zone_type_to_id[zone_type]
        num_cells = len(group)
        meta = self.zone_meta[zone_id]
        total_living = getattr(meta, "total_living_area", float("nan"))
        if total_living and not np.isnan(total_living) and total_living > 0:
            living_density = np.log1p(float(total_living) / num_cells)
            living_prompt = (living_density - self.living_mean) / max(self.living_std, 1e-6)
            living_mask = 1.0
        else:
            living_prompt = 0.0
            living_mask = 0.0
        service_cap = _parse_service_json(getattr(meta, "service_capacities", None))
        service_prompt = np.zeros(len(self.service_types), dtype=np.float32)
        service_prompt_mask = np.zeros(len(self.service_types), dtype=np.float32)
        for i, service in enumerate(self.service_types):
            value = service_cap.get(service)
            if value is None or value <= 0:
                continue
            density = np.log1p(float(value) / num_cells)
            mean = self.capacity_mean.get(service, 0.0)
            std = self.capacity_std.get(service, 1.0)
            service_prompt[i] = (density - mean) / max(std, 1e-6)
            service_prompt_mask[i] = 1.0

        building_present = group["building_id"].notna().to_numpy()
        service_present = group["service_id"].notna().to_numpy()
        both = building_present & service_present
        cell_class = np.full(num_cells, 0, dtype=np.int64)
        cell_class[building_present & ~service_present] = 1
        cell_class[service_present & ~building_present] = 2
        cell_class[both] = 3

        living_area = group["building_living_area"].fillna(0.0).to_numpy(dtype=np.float32)
        is_living = (living_area > 0).astype(np.float32)
        is_living_mask = building_present.astype(np.float32)
        living_area_mask = building_present.astype(np.float32)
        storeys = group["building_storeys_count"].fillna(0.0).to_numpy(dtype=np.float32)
        storeys_available = group["building_storeys_count"].notna().to_numpy()
        storeys_mask = np.logical_and(storeys_available, building_present).astype(np.float32)

        service_type_series = group["service_type_name"].fillna("__none__")
        service_type_ids = np.full(num_cells, -1, dtype=np.int64)
        for idx, service in enumerate(self.service_types):
            service_type_ids[service_type_series == service] = idx
        service_type_mask = service_present.astype(np.float32)
        service_capacity = group["service_capacity"].fillna(0.0).to_numpy(dtype=np.float32)
        service_capacity_mask = service_present.astype(np.float32)

        prompt = {
            "zone_type_id": torch.tensor(zone_type_id, dtype=torch.long),
            "living_area": torch.tensor([living_prompt], dtype=torch.float32),
            "living_area_mask": torch.tensor([living_mask], dtype=torch.float32),
            "service_cap": torch.from_numpy(service_prompt),
            "service_cap_mask": torch.from_numpy(service_prompt_mask),
        }
        return ZoneExample(
            zone_id=str(zone_id),
            zone_type_id=zone_type_id,
            prompt=prompt,
            cell_class=torch.from_numpy(cell_class),
            is_living=torch.from_numpy(is_living),
            is_living_mask=torch.from_numpy(is_living_mask),
            storeys=torch.from_numpy(storeys),
            storeys_mask=torch.from_numpy(storeys_mask),
            living_area=torch.from_numpy(living_area),
            living_area_mask=torch.from_numpy(living_area_mask),
            service_type=torch.from_numpy(service_type_ids),
            service_type_mask=torch.from_numpy(service_type_mask),
            service_capacity=torch.from_numpy(service_capacity),
            service_capacity_mask=torch.from_numpy(service_capacity_mask),
            edge_distances=torch.from_numpy(distances),
        )


def collate_zone_batch(batch: Iterable[ZoneExample]) -> Dict[str, torch.Tensor]:
    """Pad and collate a batch of :class:`ZoneExample` objects."""
    batch_list = list(batch)
    max_cells = max(example.cell_class.numel() for example in batch_list)
    B = len(batch_list)

    inputs = {}
    cell_class = torch.full((B, max_cells), -1, dtype=torch.long)
    mask = torch.zeros((B, max_cells), dtype=torch.float32)
    edge_distances = torch.zeros((B, max_cells, len(DIRECTIONS)), dtype=torch.float32)
    is_living = torch.zeros((B, max_cells), dtype=torch.float32)
    is_living_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    storeys = torch.zeros((B, max_cells), dtype=torch.float32)
    storeys_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    living_area = torch.zeros((B, max_cells), dtype=torch.float32)
    living_area_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    service_type = torch.full((B, max_cells), -1, dtype=torch.long)
    service_type_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    service_capacity = torch.zeros((B, max_cells), dtype=torch.float32)
    service_capacity_mask = torch.zeros((B, max_cells), dtype=torch.float32)

    zone_type_ids = torch.tensor([ex.zone_type_id for ex in batch_list], dtype=torch.long)
    living_prompt = torch.stack([ex.prompt["living_area"] for ex in batch_list], dim=0)
    living_prompt_mask = torch.stack([ex.prompt["living_area_mask"] for ex in batch_list], dim=0)
    service_prompt = torch.stack([ex.prompt["service_cap"] for ex in batch_list], dim=0)
    service_prompt_mask = torch.stack([ex.prompt["service_cap_mask"] for ex in batch_list], dim=0)

    for i, example in enumerate(batch_list):
        length = example.cell_class.numel()
        cell_class[i, :length] = example.cell_class
        mask[i, :length] = 1.0
        edge_distances[i, :length] = example.edge_distances
        is_living[i, :length] = example.is_living
        is_living_mask[i, :length] = example.is_living_mask
        storeys[i, :length] = example.storeys
        storeys_mask[i, :length] = example.storeys_mask
        living_area[i, :length] = example.living_area
        living_area_mask[i, :length] = example.living_area_mask
        service_type[i, :length] = example.service_type
        service_type_mask[i, :length] = example.service_type_mask
        service_capacity[i, :length] = example.service_capacity
        service_capacity_mask[i, :length] = example.service_capacity_mask

    inputs.update({
        "cell_class": cell_class,
        "sequence_mask": mask,
        "edge_distances": edge_distances,
        "is_living": is_living,
        "is_living_mask": is_living_mask,
        "storeys": storeys,
        "storeys_mask": storeys_mask,
        "living_area": living_area,
        "living_area_mask": living_area_mask,
        "service_type": service_type,
        "service_type_mask": service_type_mask,
        "service_capacity": service_capacity,
        "service_capacity_mask": service_capacity_mask,
        "zone_type_ids": zone_type_ids,
        "living_prompt": living_prompt,
        "living_prompt_mask": living_prompt_mask,
        "service_prompt": service_prompt,
        "service_prompt_mask": service_prompt_mask,
        "num_cells": mask.sum(dim=1),
    })
    return inputs
