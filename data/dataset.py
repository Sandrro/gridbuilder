"""Dataset utilities for the autoregressive grid Transformer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

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
    coords: torch.FloatTensor
    cell_class: torch.LongTensor
    is_living: torch.FloatTensor
    is_living_mask: torch.FloatTensor
    storeys: torch.FloatTensor
    storeys_mask: torch.FloatTensor
    living_area: torch.FloatTensor
    living_area_mask: torch.FloatTensor
    service_presence: torch.FloatTensor
    service_presence_mask: torch.FloatTensor
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


def _normalize_coordinates(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Normalize integer grid coordinates to the ``[0, 1]`` range per axis."""

    row_min = float(rows.min()) if rows.size else 0.0
    row_max = float(rows.max()) if rows.size else 0.0
    col_min = float(cols.min()) if cols.size else 0.0
    col_max = float(cols.max()) if cols.size else 0.0

    row_range = row_max - row_min
    col_range = col_max - col_min

    if row_range <= 0:
        row_range = 1.0
    if col_range <= 0:
        col_range = 1.0

    norm_rows = (rows.astype(np.float32) - row_min) / row_range
    norm_cols = (cols.astype(np.float32) - col_min) / col_range
    return np.stack([norm_rows, norm_cols], axis=1)


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


def _read_parquet(path: str, *, storage_options: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Read a Parquet file with a clearer error message when engines are missing."""

    try:
        return pd.read_parquet(path, storage_options=storage_options)
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "Reading Parquet files requires the 'pyarrow' dependency. "
            "Install it with `pip install pyarrow` and retry."
        ) from exc


class GridDataset(Dataset):
    """PyTorch dataset that yields :class:`ZoneExample` objects."""

    def __init__(
        self,
        grid_path: str,
        descriptions_path: str,
        *,
        min_cells: int = 16,
        max_context: int | None = None,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        storage_options = {"token": hf_token} if hf_token else None
        self.grid_df = _read_parquet(grid_path, storage_options=storage_options)
        self.desc_df = _read_parquet(descriptions_path, storage_options=storage_options)
        self.min_cells = min_cells
        self.max_context = max_context

        self.zone_groups = self.grid_df.groupby("zone_id")
        self.zone_meta = {row.zone_id: row for row in self.desc_df.itertuples(index=False)}

        self.zone_ids: List[str] = []
        self.service_types: List[str] = []
        self.service_type_to_id: Dict[str, int] = {}
        self.zone_type_to_id: Dict[str, int] = {}
        self.id_to_zone_type: Dict[int, str] = {}
        self.service_type_to_id: Dict[str, int] = {}

        self.service_presence_col = next(
            (
                col
                for col in (
                    "service_types_json",
                    "service_types",
                    "service_presence_json",
                    "service_presence",
                )
                if col in self.grid_df.columns
            ),
            None,
        )
        self.service_capacity_col = next(
            (
                col
                for col in (
                    "service_capacity_distribution",
                    "service_capacities_json",
                    "service_capacity_json",
                    "service_capacity_by_type",
                )
                if col in self.grid_df.columns
            ),
            None,
        )

        self._prepare_vocabularies()
        self._collect_normalization_stats()

    def _prepare_vocabularies(self) -> None:
        service_types: set[str] = set()
        for raw in self.grid_df.get("service_types_json", pd.Series(dtype=object)).dropna():
            for name in _parse_service_json(raw).keys():
                service_types.add(str(name))
        for raw in self.grid_df.get("service_capacity_json", pd.Series(dtype=object)).dropna():
            for name in _parse_service_json(raw).keys():
                service_types.add(str(name))
        if "service_type_name" in self.grid_df.columns:
            for name in self.grid_df["service_type_name"].dropna().unique():
                service_types.add(str(name))
        json_columns = [col for col in (self.service_presence_col, self.service_capacity_col) if col]
        for col in json_columns:
            series = self.grid_df[col]
            for raw in series.dropna().tolist():
                for service in _parse_service_json(raw).keys():
                    service_types.add(service)
        self.service_types = sorted(service_types)
        self.service_type_to_id = {service: idx for idx, service in enumerate(self.service_types)}
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
        coords = _normalize_coordinates(rows, cols)

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
        if "service_types_json" in group:
            service_counts_series = group["service_types_json"].apply(_parse_service_json)
        else:
            service_counts_series = pd.Series([{} for _ in range(num_cells)], index=group.index)
        if "service_capacity_json" in group:
            service_capacity_series = group["service_capacity_json"].apply(_parse_service_json)
        else:
            service_capacity_series = pd.Series([{} for _ in range(num_cells)], index=group.index)

        service_counts_list = service_counts_series.tolist()
        service_capacity_list = service_capacity_series.tolist()
        service_present = np.array([bool(entry) for entry in service_counts_list], dtype=bool)
        both = building_present & service_present
        cell_class = np.full(num_cells, 0, dtype=np.int64)
        cell_class[building_present & ~service_present] = 1
        cell_class[service_present & ~building_present] = 2
        cell_class[both] = 3

        service_present = np.zeros(num_cells, dtype=bool)
        living_area = group["building_living_area"].fillna(0.0).to_numpy(dtype=np.float32)
        is_living = (living_area > 0).astype(np.float32)
        is_living_mask = building_present.astype(np.float32)
        living_area_mask = building_present.astype(np.float32)
        storeys = group["building_storeys_count"].fillna(0.0).to_numpy(dtype=np.float32)
        storeys_available = group["building_storeys_count"].notna().to_numpy()
        storeys_mask = np.logical_and(storeys_available, building_present).astype(np.float32)

        service_type_ids = np.full(num_cells, -1, dtype=np.int64)
        service_capacity = np.zeros(num_cells, dtype=np.float32)
        for idx_cell, (counts, capacities) in enumerate(zip(service_counts_list, service_capacity_list)):
            primary_type = "__none__"
            if capacities:
                primary_type = max(capacities.items(), key=lambda item: float(item[1]))[0]
                service_capacity[idx_cell] = float(sum(float(v) for v in capacities.values()))
            elif counts:
                primary_type = max(counts.items(), key=lambda item: float(item[1]))[0]
            mapped = self.service_type_to_id.get(str(primary_type))
            if mapped is not None:
                service_type_ids[idx_cell] = mapped
        service_type_mask = service_present.astype(np.float32)
        service_capacity_mask = (service_capacity > 0).astype(np.float32)
        service_presence = np.zeros((num_cells, len(self.service_types)), dtype=np.float32)
        service_presence_mask = np.zeros_like(service_presence)
        service_capacity = np.zeros((num_cells, len(self.service_types)), dtype=np.float32)
        service_capacity_mask = np.zeros_like(service_capacity)

        def _update_presence(idx: int, mapping: Dict[str, float]) -> bool:
            has_service = False
            for service_name, value in mapping.items():
                if service_name not in self.service_type_to_id:
                    continue
                if value <= 0:
                    continue
                col_idx = self.service_type_to_id[service_name]
                service_presence[idx, col_idx] = 1.0
                has_service = True
            return has_service

        def _update_capacity(idx: int, mapping: Dict[str, float]) -> bool:
            has_service = False
            for service_name, value in mapping.items():
                if service_name not in self.service_type_to_id:
                    continue
                if value <= 0:
                    continue
                col_idx = self.service_type_to_id[service_name]
                service_capacity[idx, col_idx] = float(value)
                service_presence[idx, col_idx] = 1.0
                has_service = True
            return has_service

        for cell_idx in range(num_cells):
            presence_map: Dict[str, float] = {}
            if self.service_presence_col and self.service_presence_col in group.columns:
                presence_map = _parse_service_json(group.iloc[cell_idx][self.service_presence_col])
            elif "service_type_name" in group.columns:
                name = group.iloc[cell_idx]["service_type_name"]
                if pd.notna(name):
                    presence_map = {str(name): 1.0}

            capacity_map: Dict[str, float] = {}
            if self.service_capacity_col and self.service_capacity_col in group.columns:
                capacity_map = _parse_service_json(group.iloc[cell_idx][self.service_capacity_col])
            elif "service_capacity" in group.columns and "service_type_name" in group.columns:
                cap_value = group.iloc[cell_idx]["service_capacity"]
                name = group.iloc[cell_idx]["service_type_name"]
                if pd.notna(cap_value) and pd.notna(name):
                    capacity_map = {str(name): float(cap_value)}

            has_service = False
            if presence_map:
                has_service = _update_presence(cell_idx, presence_map)
            if capacity_map:
                has_service = _update_capacity(cell_idx, capacity_map) or has_service

            if has_service:
                service_presence_mask[cell_idx, :] = 1.0
                service_capacity_mask[cell_idx, :] = service_presence[cell_idx]
                service_present[cell_idx] = True

        both = building_present & service_present
        cell_class = np.full(num_cells, 0, dtype=np.int64)
        cell_class[building_present & ~service_present] = 1
        cell_class[service_present & ~building_present] = 2
        cell_class[both] = 3

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
            coords=torch.from_numpy(coords.astype(np.float32)),
            cell_class=torch.from_numpy(cell_class),
            is_living=torch.from_numpy(is_living),
            is_living_mask=torch.from_numpy(is_living_mask),
            storeys=torch.from_numpy(storeys),
            storeys_mask=torch.from_numpy(storeys_mask),
            living_area=torch.from_numpy(living_area),
            living_area_mask=torch.from_numpy(living_area_mask),
            service_presence=torch.from_numpy(service_presence),
            service_presence_mask=torch.from_numpy(service_presence_mask),
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
    coords = torch.zeros((B, max_cells, 2), dtype=torch.float32)
    is_living = torch.zeros((B, max_cells), dtype=torch.float32)
    is_living_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    storeys = torch.zeros((B, max_cells), dtype=torch.float32)
    storeys_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    living_area = torch.zeros((B, max_cells), dtype=torch.float32)
    living_area_mask = torch.zeros((B, max_cells), dtype=torch.float32)
    service_dim = batch_list[0].service_presence.size(1) if batch_list else 0
    service_presence = torch.zeros((B, max_cells, service_dim), dtype=torch.float32)
    service_presence_mask = torch.zeros((B, max_cells, service_dim), dtype=torch.float32)
    service_capacity = torch.zeros((B, max_cells, service_dim), dtype=torch.float32)
    service_capacity_mask = torch.zeros((B, max_cells, service_dim), dtype=torch.float32)

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
        coords[i, :length] = example.coords
        is_living[i, :length] = example.is_living
        is_living_mask[i, :length] = example.is_living_mask
        storeys[i, :length] = example.storeys
        storeys_mask[i, :length] = example.storeys_mask
        living_area[i, :length] = example.living_area
        living_area_mask[i, :length] = example.living_area_mask
        service_presence[i, :length] = example.service_presence
        service_presence_mask[i, :length] = example.service_presence_mask
        service_capacity[i, :length] = example.service_capacity
        service_capacity_mask[i, :length] = example.service_capacity_mask

    inputs.update({
        "cell_class": cell_class,
        "sequence_mask": mask,
        "edge_distances": edge_distances,
        "cell_coords": coords,
        "is_living": is_living,
        "is_living_mask": is_living_mask,
        "storeys": storeys,
        "storeys_mask": storeys_mask,
        "living_area": living_area,
        "living_area_mask": living_area_mask,
        "service_presence": service_presence,
        "service_presence_mask": service_presence_mask,
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
