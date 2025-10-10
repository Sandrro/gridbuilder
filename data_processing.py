"""Tools for preparing zone descriptions and uniform grids from GeoJSON inputs.

The script expects three GeoJSON inputs (zones, buildings, services) and
produces two Parquet datasets:

* descriptions.parquet – aggregated information for each zone that contains at
  least one building.
* grid_cells.parquet – per-cell information for a 15 m (configurable) square
  grid fitted to every zone.

Usage::

    python data_processing.py zones.geojson buildings.geojson services.geojson output_dir \
        [--grid-size 15.0] [--workers 4]

The resulting Parquet files are written inside ``output_dir``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - hard failure without pyarrow
    raise ImportError("pyarrow is required for Parquet output. Please install pyarrow.") from exc
from shapely import wkb
from shapely.geometry import box
from shapely.prepared import prep
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


@dataclass
class BuildingRecord:
    building_id: object
    geometry: object
    bounds: Tuple[float, float, float, float]
    storeys_count: Optional[float]
    living_area_in_zone: Optional[float]
    building_year: Optional[float]


@dataclass
class ServiceRecord:
    service_id: object
    geometry: object
    bounds: Tuple[float, float, float, float]
    is_point: bool
    service_type: Optional[str]
    capacity: Optional[float]

_GLOBAL_BUILDINGS: Optional[gpd.GeoDataFrame] = None
_GLOBAL_SERVICES: Optional[gpd.GeoDataFrame] = None
_GLOBAL_GRID_SIZE: float = 0.0


def _init_worker(buildings_df: gpd.GeoDataFrame, services_df: gpd.GeoDataFrame, grid_size: float) -> None:
    global _GLOBAL_BUILDINGS, _GLOBAL_SERVICES, _GLOBAL_GRID_SIZE

    _GLOBAL_BUILDINGS = buildings_df
    _GLOBAL_SERVICES = services_df
    _GLOBAL_GRID_SIZE = grid_size


def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf = gdf.copy()
        gdf.set_crs(target_crs, allow_override=True, inplace=True)
        return gdf
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame is missing CRS information")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def _bbox_intersects(bounds_a: Tuple[float, float, float, float],
                     bounds_b: Tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = bounds_a
    minx2, miny2, maxx2, maxy2 = bounds_b
    return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)


def _hash_seed(value: object) -> int:
    import hashlib

    raw = str(value).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def _build_strtree(records: Iterable[object]) -> Tuple[Optional[STRtree], Dict[int, object]]:
    geometries = [record.geometry for record in records if getattr(record, "geometry", None) is not None]
    if not geometries:
        return None, {}
    tree = STRtree(geometries)
    lookup = {id(geom): record for geom, record in zip(geometries, records)}
    return tree, lookup


def _query_tree_records(tree: Optional[STRtree], lookup: Dict[int, object], fallback: Sequence[object], geom) -> Sequence[object]:
    if tree is None:
        return fallback
    try:
        candidates = tree.query(geom, predicate="intersects")
    except TypeError:  # Older shapely
        candidates = tree.query(geom)
    candidates = list(candidates)
    if not candidates:
        return ()
    seen: set[int] = set()
    results: List[object] = []
    for candidate in candidates:
        record = lookup.get(id(candidate))
        if record is None:
            continue
        marker = id(record)
        if marker in seen:
            continue
        seen.add(marker)
        results.append(record)
    return results


def _write_parquet_chunk(writer: Optional[pq.ParquetWriter], path: Path, records: Sequence[Dict[str, object]]) -> Optional[pq.ParquetWriter]:
    if not records:
        return writer
    df = pd.DataFrame(list(records))
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _prepare_buildings(zone_geom, zone_prepared, buildings_df: gpd.GeoDataFrame) -> Tuple[List[BuildingRecord], float, Optional[float], Optional[float]]:
    zone_bounds = zone_geom.bounds
    records: List[BuildingRecord] = []
    storeys_values: List[float] = []
    coverage_area = 0.0

    for building in buildings_df.itertuples():
        geom = building.geometry
        if geom is None or geom.is_empty:
            continue
        if not _bbox_intersects(zone_bounds, geom.bounds):
            continue
        if zone_prepared.contains(geom):
            intersection = geom
        else:
            intersection = geom.intersection(zone_geom)
        if intersection.is_empty:
            continue
        area_in_zone = intersection.area
        if area_in_zone <= 0.0:
            continue
        coverage_area += area_in_zone

        living_area_value = getattr(building, "living_area", None)
        total_geom_area = geom.area
        living_area_in_zone: Optional[float]
        if living_area_value is None or (isinstance(living_area_value, float) and math.isnan(living_area_value)):
            living_area_in_zone = None
        else:
            if total_geom_area > 0:
                living_area_in_zone = living_area_value * (area_in_zone / total_geom_area)
            else:
                living_area_in_zone = living_area_value

        storeys = getattr(building, "storeys_count", None)
        if storeys is not None and not (isinstance(storeys, float) and math.isnan(storeys)):
            storeys_values.append(storeys)

        year = getattr(building, "building_year", None)
        if year is not None and isinstance(year, float) and math.isnan(year):
            year = None

        building_identifier = getattr(building, "building_id", None)
        if building_identifier is None:
            building_identifier = getattr(building, "physical_object_id", None)
        if building_identifier is None:
            building_identifier = getattr(building, "id", None)

        records.append(
            BuildingRecord(
                building_id=building_identifier,
                geometry=intersection,
                bounds=intersection.bounds,
                storeys_count=None if storeys is None or (isinstance(storeys, float) and math.isnan(storeys)) else storeys,
                living_area_in_zone=living_area_in_zone,
                building_year=year,
            )
        )

    storeys_min = min(storeys_values) if storeys_values else None
    storeys_max = max(storeys_values) if storeys_values else None

    return records, coverage_area, storeys_min, storeys_max


def _prepare_services(zone_geom, zone_prepared, services_df: gpd.GeoDataFrame) -> Tuple[List[ServiceRecord], Dict[str, Dict[str, float]]]:
    zone_bounds = zone_geom.bounds
    summary: Dict[str, Dict[str, float]] = {}
    records: List[ServiceRecord] = []

    for service in services_df.itertuples():
        geom = service.geometry
        if geom is None or geom.is_empty:
            continue
        if not _bbox_intersects(zone_bounds, geom.bounds):
            continue
        if zone_prepared.contains(geom):
            intersection = geom
        else:
            intersection = geom.intersection(zone_geom)
        if intersection.is_empty:
            continue
        service_type = getattr(service, "service_type_name", None)
        service_capacity = getattr(service, "capacity", None)
        entry = summary.setdefault(service_type or "unknown", {"count": 0, "capacity": 0.0, "capacity_valid": False})
        entry["count"] += 1
        if service_capacity is not None and not (isinstance(service_capacity, float) and math.isnan(service_capacity)):
            entry["capacity"] += float(service_capacity)
            entry["capacity_valid"] = True

        is_point = intersection.geom_type in {"Point", "MultiPoint"}
        service_identifier = getattr(service, "service_id", None)
        if service_identifier is None:
            service_identifier = getattr(service, "object_geometry_id", None)
        if service_identifier is None:
            service_identifier = getattr(service, "id", None)

        records.append(
            ServiceRecord(
                service_id=service_identifier,
                geometry=intersection,
                bounds=intersection.bounds,
                is_point=is_point,
                service_type=service_type,
                capacity=None if service_capacity is None or (isinstance(service_capacity, float) and math.isnan(service_capacity)) else float(service_capacity),
            )
        )

    for stats in summary.values():
        if not stats.pop("capacity_valid"):
            stats["capacity"] = None

    return records, summary


def _assign_ring_order(cells: List[Dict[str, object]], zone_geom) -> None:
    if not cells:
        return

    zone_boundary = zone_geom.boundary
    index_map = {(cell["row"], cell["col"]): idx for idx, cell in enumerate(cells)}
    queue: deque[int] = deque()

    for idx, cell in enumerate(cells):
        intersection = cell["intersection"]
        if intersection.is_empty:
            continue
        if intersection.touches(zone_boundary) or math.isclose(intersection.area, cell["geometry"].area, rel_tol=1e-9, abs_tol=1e-9) and not zone_geom.contains(cell["geometry"].centroid):
            cell["ring_index"] = 0
            queue.append(idx)

    if not queue:
        # If every cell is interior (e.g., a single cell), choose all as ring 0.
        for idx in range(len(cells)):
            cells[idx]["ring_index"] = 0
            queue.append(idx)

    while queue:
        current_idx = queue.popleft()
        current_cell = cells[current_idx]
        row, col = current_cell["row"], current_cell["col"]
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbor_idx = index_map.get(neighbor)
            if neighbor_idx is None:
                continue
            neighbor_cell = cells[neighbor_idx]
            if neighbor_cell["ring_index"] is not None:
                continue
            neighbor_cell["ring_index"] = current_cell["ring_index"] + 1
            queue.append(neighbor_idx)

    rings: Dict[int, List[int]] = defaultdict(list)
    for idx, cell in enumerate(cells):
        ring = cell["ring_index"] if cell["ring_index"] is not None else 0
        rings[ring].append(idx)

    for ring, idxs in rings.items():
        for order, cell_idx in enumerate(sorted(idxs, key=lambda i: (cells[i]["row"], cells[i]["col"]))):
            cells[cell_idx]["ring_order"] = order


def _distribute_values(cells: List[Dict[str, object]], building_lookup: Dict[object, BuildingRecord], service_lookup: Dict[object, ServiceRecord]) -> None:
    building_cells: Dict[object, List[int]] = defaultdict(list)
    service_cells: Dict[object, List[int]] = defaultdict(list)

    for idx, cell in enumerate(cells):
        b_id = cell.get("building_id")
        if b_id is not None:
            building_cells[b_id].append(idx)
        s_id = cell.get("service_id")
        if s_id is not None:
            service_cells[s_id].append(idx)

    for b_id, idxs in building_cells.items():
        building = building_lookup.get(b_id)
        if building is None:
            continue
        for idx in idxs:
            cells[idx]["building_storeys_count"] = building.storeys_count
            cells[idx]["building_year"] = building.building_year
        living = building.living_area_in_zone
        if living is not None:
            share = living / len(idxs) if idxs else None
            if share is not None:
                for idx in idxs:
                    cells[idx]["building_living_area"] = share

    for s_id, idxs in service_cells.items():
        service = service_lookup.get(s_id)
        if service is None:
            continue
        for idx in idxs:
            cells[idx]["service_type_name"] = service.service_type
        capacity = service.capacity
        if capacity is not None:
            share = capacity / len(idxs) if idxs else None
            if share is not None:
                for idx in idxs:
                    cells[idx]["service_capacity"] = share


def _process_zone(zone_row, buildings_df: gpd.GeoDataFrame, services_df: gpd.GeoDataFrame, cell_size: float) -> Optional[Tuple[Dict[str, object], List[Dict[str, object]]]]:
    zone_geom = zone_row.geometry
    if zone_geom is None or zone_geom.is_empty:
        return None

    zone_area = zone_geom.area
    if zone_area <= 0:
        return None

    zone_id = getattr(zone_row, "functional_zone_id", getattr(zone_row, "id", None))
    rng = random.Random(_hash_seed(zone_id))
    zone_prepared = prep(zone_geom)

    buildings_candidates = buildings_df
    if hasattr(buildings_df, "sindex") and buildings_df.sindex is not None:
        candidate_idx = buildings_df.sindex.query(zone_geom, predicate="intersects")
        buildings_candidates = buildings_df.iloc[np.asarray(candidate_idx)]

    building_records, coverage_area, storeys_min, storeys_max = _prepare_buildings(zone_geom, zone_prepared, buildings_candidates)
    if not building_records:
        return None

    building_tree, building_lookup_by_geom = _build_strtree(building_records)

    services_candidates = services_df
    if hasattr(services_df, "sindex") and services_df.sindex is not None:
        candidate_idx = services_df.sindex.query(zone_geom, predicate="intersects")
        services_candidates = services_df.iloc[np.asarray(candidate_idx)]

    service_records, service_summary = _prepare_services(zone_geom, zone_prepared, services_candidates)
    service_tree, service_lookup_by_geom = _build_strtree(service_records)

    minx, miny, maxx, maxy = zone_geom.bounds
    start_x = math.floor(minx / cell_size) * cell_size
    start_y = math.floor(miny / cell_size) * cell_size
    end_x = math.ceil(maxx / cell_size) * cell_size
    end_y = math.ceil(maxy / cell_size) * cell_size

    cols = max(1, int(round((end_x - start_x) / cell_size)))
    rows = max(1, int(round((end_y - start_y) / cell_size)))

    cells: List[Dict[str, object]] = []
    zone_bounds = zone_geom.bounds
    cell_area = cell_size * cell_size

    building_lookup = {record.building_id: record for record in building_records if record.building_id is not None}
    service_lookup = {record.service_id: record for record in service_records if record.service_id is not None}

    for row_idx in range(rows):
        y0 = start_y + row_idx * cell_size
        y1 = y0 + cell_size
        for col_idx in range(cols):
            x0 = start_x + col_idx * cell_size
            x1 = x0 + cell_size
            cell_geom = box(x0, y0, x1, y1)
            if not _bbox_intersects(cell_geom.bounds, zone_bounds):
                continue
            if not zone_prepared.intersects(cell_geom):
                continue
            if zone_prepared.contains(cell_geom):
                intersection = cell_geom
            else:
                intersection = cell_geom.intersection(zone_geom)
            if intersection.is_empty:
                continue

            cell_record: Dict[str, object] = {
                "zone_id": zone_id,
                "row": row_idx,
                "col": col_idx,
                "geometry": cell_geom,
                "intersection": intersection,
                "ring_index": None,
                "ring_order": None,
                "building_id": None,
                "building_storeys_count": None,
                "building_living_area": None,
                "building_year": None,
                "service_id": None,
                "service_type_name": None,
                "service_capacity": None,
            }

            # Building assignment
            best_building = None
            best_coverage = 0.0
            if building_records:
                candidate_buildings = _query_tree_records(building_tree, building_lookup_by_geom, building_records, cell_geom)
                ratios: List[Tuple[BuildingRecord, float]] = []
                for building in candidate_buildings:
                    if not _bbox_intersects(building.bounds, cell_geom.bounds):
                        continue
                    coverage = building.geometry.intersection(cell_geom).area
                    if coverage <= 0:
                        continue
                    ratio = coverage / cell_area if cell_area > 0 else 0.0
                    if ratio <= 0:
                        continue
                    ratios.append((building, ratio))

                if ratios:
                    max_ratio = max(ratio for _, ratio in ratios)
                    if max_ratio > 0.5 + 1e-9:
                        top_candidates = [rec for rec, ratio in ratios if math.isclose(ratio, max_ratio, rel_tol=1e-9, abs_tol=1e-9)]
                        best_building = rng.choice(top_candidates) if len(top_candidates) > 1 else top_candidates[0]
                        cell_record["building_id"] = best_building.building_id
                    elif math.isclose(max_ratio, 0.5, rel_tol=1e-9, abs_tol=1e-9):
                        tied_half = [rec for rec, ratio in ratios if math.isclose(ratio, 0.5, rel_tol=1e-9, abs_tol=1e-9)]
                        if tied_half:
                            chosen = rng.choice(tied_half) if len(tied_half) > 1 else tied_half[0]
                            cell_record["building_id"] = chosen.building_id

            # Service assignment
            if service_records:
                candidate_services = _query_tree_records(service_tree, service_lookup_by_geom, service_records, cell_geom)
                service_ratios: List[Tuple[ServiceRecord, float]] = []
                for service in candidate_services:
                    if not _bbox_intersects(service.bounds, cell_geom.bounds):
                        continue
                    if service.is_point:
                        if service.geometry.geom_type == "Point":
                            coverage_ratio = 1.0 if cell_geom.contains(service.geometry) else 0.0
                        else:
                            points = list(service.geometry.geoms)
                            coverage_ratio = 1.0 if any(cell_geom.contains(pt) for pt in points) else 0.0
                    else:
                        coverage = service.geometry.intersection(cell_geom).area
                        coverage_ratio = coverage / cell_area if cell_area > 0 else 0.0
                    if coverage_ratio <= 0:
                        continue
                    service_ratios.append((service, coverage_ratio))

                if service_ratios:
                    max_ratio = max(ratio for _, ratio in service_ratios)
                    if max_ratio > 0.5 + 1e-9:
                        top_candidates = [rec for rec, ratio in service_ratios if math.isclose(ratio, max_ratio, rel_tol=1e-9, abs_tol=1e-9)]
                        best_service = rng.choice(top_candidates) if len(top_candidates) > 1 else top_candidates[0]
                        cell_record["service_id"] = best_service.service_id
                    elif math.isclose(max_ratio, 0.5, rel_tol=1e-9, abs_tol=1e-9):
                        tied_half = [rec for rec, ratio in service_ratios if math.isclose(ratio, 0.5, rel_tol=1e-9, abs_tol=1e-9)]
                        if tied_half:
                            chosen_service = rng.choice(tied_half) if len(tied_half) > 1 else tied_half[0]
                            cell_record["service_id"] = chosen_service.service_id

            cells.append(cell_record)

    if not cells:
        return None

    _assign_ring_order(cells, zone_geom)
    _distribute_values(cells, building_lookup, service_lookup)

    for cell in cells:
        cell.pop("intersection", None)

    description_record = {
        "zone_id": zone_id,
        "zone_type": getattr(zone_row, "functional_zone_type_name", None),
        "total_living_area": np.nan,
        "building_coverage_ratio": coverage_area / zone_area if zone_area > 0 else np.nan,
        "storeys_min": np.nan if storeys_min is None else float(storeys_min),
        "storeys_max": np.nan if storeys_max is None else float(storeys_max),
        "service_counts": json.dumps({k: v["count"] for k, v in service_summary.items()}, ensure_ascii=False),
        "service_capacities": json.dumps({k: v["capacity"] for k, v in service_summary.items()}, ensure_ascii=False),
    }

    living_values = [record.living_area_in_zone for record in building_records if record.living_area_in_zone is not None]
    if living_values:
        description_record["total_living_area"] = float(sum(living_values))

    grid_records: List[Dict[str, object]] = []
    for idx, cell in enumerate(cells):
        geometry_bytes = wkb.dumps(cell["geometry"])
        grid_records.append(
            {
                "zone_id": zone_id,
                "cell_id": f"{zone_id}_{idx}",
                "row": cell["row"],
                "col": cell["col"],
                "ring_index": cell.get("ring_index"),
                "ring_order": cell.get("ring_order"),
                "building_id": cell.get("building_id"),
                "building_storeys_count": cell.get("building_storeys_count"),
                "building_living_area": cell.get("building_living_area"),
                "building_year": cell.get("building_year"),
                "service_id": cell.get("service_id"),
                "service_type_name": cell.get("service_type_name"),
                "service_capacity": cell.get("service_capacity"),
                "geometry": geometry_bytes,
            }
        )

    return description_record, grid_records

def _process_zone_with_globals(zone_row) -> Optional[Tuple[Dict[str, object], List[Dict[str, object]]]]:
    if _GLOBAL_BUILDINGS is None or _GLOBAL_SERVICES is None:
        raise RuntimeError("Worker is not initialised with shared data")

    return _process_zone(zone_row, _GLOBAL_BUILDINGS, _GLOBAL_SERVICES, _GLOBAL_GRID_SIZE)


def process(zones_path: Path, buildings_path: Path, services_path: Path, output_dir: Path,
            grid_size: float = 15.0, workers: int = 1) -> None:
    zones = gpd.read_file(zones_path)
    buildings = gpd.read_file(buildings_path)
    services = gpd.read_file(services_path)

    target_crs = "EPSG:3857"
    zones = _ensure_crs(zones, target_crs)
    buildings = _ensure_crs(buildings, target_crs)
    services = _ensure_crs(services, target_crs)

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(zones.itertuples(index=False))

    progress_bar = None
    total_zones = len(tasks)
    if tqdm is not None and total_zones:
        progress_bar = tqdm(total=total_zones, desc="Zones", unit="zone")

    descriptions_path = output_dir / "descriptions.parquet"
    grid_path = output_dir / "grid_cells.parquet"
    descriptions_writer: Optional[pq.ParquetWriter] = None
    grid_writer: Optional[pq.ParquetWriter] = None
    zones_with_buildings = 0

    try:
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(buildings, services, grid_size),
            ) as executor:
                futures = [
                    executor.submit(_process_zone_with_globals, zone_row)
                    for zone_row in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if progress_bar is not None:
                        progress_bar.update(1)
                    if result is None:
                        continue
                    description_record, grid_record = result
                    descriptions_writer = _write_parquet_chunk(descriptions_writer, descriptions_path, [description_record])
                    grid_writer = _write_parquet_chunk(grid_writer, grid_path, grid_record)
                    zones_with_buildings += 1
        else:
            for zone_row in tasks:
                result = _process_zone(zone_row, buildings, services, grid_size)
                if progress_bar is not None:
                    progress_bar.update(1)
                if result is None:
                    continue
                description_record, grid_record = result
                descriptions_writer = _write_parquet_chunk(descriptions_writer, descriptions_path, [description_record])
                grid_writer = _write_parquet_chunk(grid_writer, grid_path, grid_record)
                zones_with_buildings += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if zones_with_buildings == 0:
        raise RuntimeError("No zones contained buildings – nothing to write")

    if descriptions_writer is not None:
        descriptions_writer.close()
    if grid_writer is not None:
        grid_writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare zone descriptions and grids from GeoJSON inputs.")
    parser.add_argument("zones", type=Path, help="Path to zones GeoJSON file")
    parser.add_argument("buildings", type=Path, help="Path to buildings GeoJSON file")
    parser.add_argument("services", type=Path, help="Path to services GeoJSON file")
    parser.add_argument("output_dir", type=Path, help="Directory where Parquet outputs will be stored")
    parser.add_argument("--grid-size", type=float, default=15.0, help="Grid cell size in metres (default: 15)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for zone processing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    process(args.zones, args.buildings, args.services, args.output_dir,
            grid_size=args.grid_size, workers=args.workers)


if __name__ == "__main__":
    main()