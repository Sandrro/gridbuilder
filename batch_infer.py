#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch driver for autoregressive grid Transformer inference over multiple blocks
using infer.py parallel batch mode.

- Reads blocks GeoJSON with property 'zone' (functional zone type).
- Reads budgets JSON: { zone_type: { "living_area": float, "services": {...} }, ... }
  Budget is applied independently to every block of that zone type.

Pipeline:
  1) For each block: choose UTM, build grid (cell_size m), compute ring_index/ring_order.
  2) Save grid (Parquet, WKB hex=False) + per-block budget JSON.
  3) Build a manifest for infer.py: [{"grid","zone_id","zone_type","budget_json","out_parquet"}...]
  4) Call infer.py once with --batch-manifest and --max-parallel (logs stream live).
  5) Join predictions back with geometries and write a single GeoJSON.

Output:
  - GeoJSON with predicted grid cells (properties include infer.py outputs).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely import to_wkb

# --------------------------- CLI --------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Model/infer params (required)
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pt/.pth)")
    p.add_argument("--vocab", required=True, help="Path to vocab.json")
    p.add_argument("--config", required=True, help="Path to inference_config.json")
    p.add_argument("--infer-script", default="infer.py", help="Path to parallel-capable inference script")

    # Data & budgets
    p.add_argument("--blocks-geojson", required=True, help="GeoJSON with blocks (polygons), property 'zone' holds zone type")
    p.add_argument("--budgets-json", required=True, help="JSON mapping zone_type -> {living_area, services{...}}")

    # Grid & output
    p.add_argument("--cell-size", type=float, default=15.0, help="Cell size in meters for grid generation")
    p.add_argument("--out-geojson", required=True, help="Path to output GeoJSON with predicted grid cells")

    # Sampling knobs passthrough
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--beam", type=int, default=1)

    # infer.py parallel mode
    p.add_argument("--max-parallel", type=int, default=4, help="Batch size for zones processed in parallel")
    p.add_argument("--infer-no-progress", action="store_true", help="Pass --no-progress to infer.py")
    p.add_argument("--infer-log-level", default="INFO", help="Pass --log-level to infer.py")
    p.add_argument("--infer-log-file", default=None, help="Optional path for infer.py --log-file (single file)")

    # Runtime
    p.add_argument("--tmpdir", default=None, help="Directory for temporary artifacts (default: mkdtemp)")
    p.add_argument("--keep-temp", action="store_true", help="Do not delete temporary files")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# --------------------------- Helpers -----------------------------

def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr)

def ensure_lonlat(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy projected to EPSG:4326 (lon/lat). If no CRS, assume EPSG:4326."""
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    s = str(gdf.crs).lower()
    if s.endswith(":4326") or s in ("epsg:4326", "wgs84", "wgs 84"):
        return gdf
    return gdf.to_crs("EPSG:4326")

def pick_utm_epsg(lon: float, lat: float) -> str:
    """Choose UTM zone EPSG. Northern: EPSG:326xx, Southern: EPSG:327xx"""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

@dataclass
class GridSpec:
    zone_id: str
    zone_type: str
    df: pd.DataFrame  # columns: cell_id,row,col,ring_index,ring_order,geometry(WKB)
    gdf_plain: gpd.GeoDataFrame
    epsg: str

def ensure_multipolygon(geom) -> MultiPolygon:
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    raise ValueError("Block geometry must be Polygon or MultiPolygon")

def build_grid_for_block(poly: MultiPolygon, cell: float) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Build a square-cell grid clipped to 'poly' in its current (metric) CRS."""
    minx, miny, maxx, maxy = poly.bounds
    nx = max(1, int(math.ceil((maxx - minx) / cell)))
    ny = max(1, int(math.ceil((maxy - miny) / cell)))

    cells, rows_cols = [], []
    for r in range(ny):
        y0 = miny + r * cell
        y1 = y0 + cell
        for c in range(nx):
            x0 = minx + c * cell
            x1 = x0 + cell
            sq = box(x0, y0, x1, y1)
            inter = sq.intersection(poly)
            if not inter.is_empty and inter.area > 0:
                cells.append(inter)
                rows_cols.append((r, c))

    if not cells:
        return pd.DataFrame(columns=["cell_id","row","col","ring_index","ring_order","geometry"]), gpd.GeoDataFrame(geometry=[], crs=None)

    gdf = gpd.GeoDataFrame(
        {"row": [rc[0] for rc in rows_cols],
         "col": [rc[1] for rc in rows_cols],
         "geometry": cells},
        crs=None
    )

    # ring_index = floor(dist_to_boundary / cell)
    boundary = poly.boundary
    centroids = gdf.geometry.centroid
    dist_to_boundary = centroids.distance(boundary)
    ring_index = np.floor(dist_to_boundary.values / max(cell, 1e-6)).astype(int)

    # ring_order by polar angle around polygon centroid
    poly_c = poly.centroid
    vx = centroids.x.values - poly_c.x
    vy = centroids.y.values - poly_c.y
    angles = np.arctan2(vy, vx)

    gdf["ring_index"] = ring_index
    gdf["angle"] = angles

    # order inside ring
    gdf["ring_order"] = 0
    for ring in np.unique(ring_index):
        mask = gdf["ring_index"] == ring
        order = np.argsort(gdf.loc[mask, "angle"].values)
        gdf.loc[mask, "ring_order"] = order

    gdf.sort_values(["ring_index", "ring_order"], inplace=True, kind="mergesort")
    gdf = gdf.reset_index(drop=True)
    gdf["cell_id"] = np.arange(len(gdf), dtype=int)

    gdf_plain = gdf.drop(columns=["angle"]).copy()
    wkb_bytes = gdf_plain.geometry.apply(lambda geom: to_wkb(geom, hex=False))
    df = pd.DataFrame({
        "cell_id": gdf_plain["cell_id"].astype(int),
        "row": gdf_plain["row"].astype(int),
        "col": gdf_plain["col"].astype(int),
        "ring_index": gdf_plain["ring_index"].astype(int),
        "ring_order": gdf_plain["ring_order"].astype(int),
        "geometry": wkb_bytes,
    })
    return df, gdf_plain

def write_grid_parquet(path: Path, zone_id: str, df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2.insert(0, "zone_id", zone_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(path, index=False)

# --------------------------- Main flow ---------------------------

def main() -> None:
    args = parse_args()
    tmp_root = Path(args.tmpdir) if args.tmpdir else Path(tempfile.mkdtemp(prefix="batch_infer_"))

    try:
        # Load blocks and budgets
        blocks_gdf = gpd.read_file(args.blocks_geojson)
        if "zone" not in blocks_gdf.columns:
            raise ValueError("Input GeoJSON must have property 'zone' for each block.")
        with open(args.budgets_json, "r", encoding="utf-8") as f:
            budgets_by_zone: Dict[str, Dict[str, Any]] = json.load(f)

        # For UTM pick we need lon/lat
        blocks_lonlat = ensure_lonlat(blocks_gdf)

        # Prepare per-block artifacts and build manifest
        manifest: List[Dict[str, Any]] = []
        zone_geom_cache: Dict[str, Tuple[gpd.GeoDataFrame, str]] = {}

        for idx, row in blocks_gdf.iterrows():
            geom = row.geometry
            if not isinstance(geom, (Polygon, MultiPolygon)):
                from shapely.geometry import shape as _shape
                geom = _shape(geom)
            block_geom = ensure_multipolygon(geom)
            zone_type = str(row["zone"])
            zone_id = str(row.get("id", idx))  # provided id or fallback index

            # UTM per block
            lonlat_centroid: Point = blocks_lonlat.geometry.iloc[idx].centroid
            epsg = pick_utm_epsg(lonlat_centroid.x, lonlat_centroid.y)

            # Project block
            block_gdf = gpd.GeoDataFrame(geometry=[block_geom], crs=blocks_gdf.crs).to_crs(epsg)
            block_poly_metric = ensure_multipolygon(block_gdf.geometry.iloc[0])

            # Build grid
            grid_df, grid_gdf_plain = build_grid_for_block(block_poly_metric, args.cell_size)
            if grid_df.empty:
                log(f"[skip] Block {zone_id}: no cells (cell-size={args.cell_size}).", args.verbose)
                continue

            # Temp paths
            work_dir = tmp_root / f"zone_{zone_id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            grid_parquet = work_dir / "grid_cells.parquet"
            pred_parquet = work_dir / "pred.parquet"
            budget_json_path = work_dir / "budget.json"

            # Persist grid parquet
            write_grid_parquet(grid_parquet, zone_id, grid_df)

            # Per-block budget
            zb = budgets_by_zone.get(zone_type, {}) or {}
            living_area = float(zb.get("living_area", 0.0) or 0.0)
            services = zb.get("services", {}) if isinstance(zb.get("services"), dict) else {}
            with open(budget_json_path, "w", encoding="utf-8") as f:
                json.dump({"living_area": living_area, "services": services}, f, ensure_ascii=False, indent=2)

            # Manifest entry required by infer.py batch mode
            manifest.append({
                "grid": str(grid_parquet),
                "zone_id": zone_id,
                "zone_type": zone_type,
                "budget_json": str(budget_json_path),
                "out_parquet": str(pred_parquet),
                # "out_geojson": can be omitted; we assemble a single GeoJSON below
            })

            # Cache metric geoms to reproject back later
            zone_geom_cache[zone_id] = (gpd.GeoDataFrame(grid_gdf_plain, geometry="geometry", crs=epsg), epsg)

            log(f"[prep] Block {zone_id}: cells={len(grid_gdf_plain)} zone={zone_type}", args.verbose)

        if not manifest:
            raise RuntimeError("No blocks prepared for inference (empty manifest).")

        # Save manifest
        manifest_path = tmp_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # Build infer.py command (parallel batch mode)
        cmd = [
            sys.executable, str(args.infer_script),
            "--model", str(args.model),
            "--vocab", str(args.vocab),
            "--config", str(args.config),
            "--batch-manifest", str(manifest_path),
            "--max-parallel", str(max(1, int(args.max_parallel))),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--beam", str(args.beam),
            "--log-level", str(args.infer_log_level),
        ]
        if args.infer_no_progress:
            cmd.append("--no-progress")
        if args.infer_log_file:
            Path(args.infer_log_file).parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--log-file", str(args.infer_log_file)]

        # Run infer.py once; stream logs live
        if args.verbose:
            sys.stderr.write("RUN: " + " ".join(cmd) + "\n")
            sys.stderr.flush()

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    sys.stderr.write(line)
                    sys.stderr.flush()
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"infer.py batch run failed with code {ret}")

        # Assemble final GeoJSON
        out_features: List[dict] = []
        for entry in manifest:
            zone_id = entry["zone_id"]
            pred_parquet = entry["out_parquet"]
            pred_df = pd.read_parquet(pred_parquet)

            geom_gdf_metric, epsg = zone_geom_cache[zone_id]
            merged = pred_df.merge(geom_gdf_metric[["cell_id", "geometry"]], on="cell_id", how="left")
            merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=epsg).to_crs(blocks_gdf.crs)

            # zone_type was not persisted in pred_df; add from manifest
            zone_type = next((m["zone_type"] for m in manifest if m["zone_id"] == zone_id), None)

            for _, prow in merged_gdf.iterrows():
                props = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in prow.items() if k != "geometry"}
                props["zone"] = zone_type
                props["zone_id"] = zone_id
                out_features.append({
                    "type": "Feature",
                    "geometry": prow.geometry.__geo_interface__,
                    "properties": props,
                })

            log(f"[ok] Block {zone_id}: merged cells={len(merged_gdf)}", args.verbose)

        out_fc = {"type": "FeatureCollection", "features": out_features}
        Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_geojson, "w", encoding="utf-8") as f:
            json.dump(out_fc, f, ensure_ascii=False)
        print(f"Saved: {args.out_geojson}")

    finally:
        if not args.keep_temp and tmp_root.exists() and (args.tmpdir is None):
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
