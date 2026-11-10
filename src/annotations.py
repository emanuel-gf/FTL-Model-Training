"""Per-tile building annotations as GeoJSON, mirroring the tiles folder.

    export_tile_annotations("data/tiles_10cm/tiles_index.csv")
    -> data/tiles_10cm/labels/<tile_id>.geojson   (one file per annotated tile)

The source is the labelme annotation of the labelled tiles (data/labelme/images, 10 cm).
Each labelme shape of a kept class is clipped to the 10 cm tile it falls in and written as a
Polygon in WGS84 (lon/lat), the same convention as the GeoJSON in outputs/.

Rules (from configs/data/buildings.yaml):
- label_map: building and building-ext become building, lote is dropped
- min_area_ratio: a shape is kept in a tile only if that share of it stays inside the tile
- a tile is annotated only if it lies fully inside an annotated labelme footprint. A tile
  without a file is not annotated, and a file with no features is annotated with no buildings.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

DATA_CONFIG = Path("configs/data/buildings.yaml")
LONLAT = "EPSG:4326"


def load_data_config() -> dict:
    return yaml.safe_load(DATA_CONFIG.read_text())


def labelled_footprints(labels_dir: str | Path, label_map: dict) -> list[dict]:
    """One entry per labelled tile (a .tif with a labelme .json): bounds, transform, CRS and kept shapes."""
    footprints = []
    for tif in sorted(Path(labels_dir).glob("*.tif")):
        json_path = tif.with_suffix(".json")
        if not json_path.exists():
            continue
        with rasterio.open(tif) as src:
            footprints.append({
                "name": tif.name,
                "stem": tif.stem,
                "bounds": tuple(src.bounds),
                "transform": src.transform,
                "crs": src.crs.to_string(),
                "shapes": _kept_shapes(json_path, label_map),
            })
    return footprints


def _kept_shapes(json_path: Path, label_map: dict) -> list[tuple[str, str, list]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = []
    for shape in data.get("shapes", []):
        cls = label_map.get(shape.get("label"))
        if cls is not None and shape.get("points"):
            shapes.append((cls, shape.get("shape_type", "polygon"), shape["points"]))
    return shapes


def inside(inner: tuple, outer: tuple, tol: float = 1e-3) -> bool:
    """True if the bounds (west, south, east, north) `inner` lie inside `outer`."""
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def tile_transform(west: float, north: float, res: float) -> Affine:
    """North-up affine of a tile: pixel (col, row) -> CRS (x, y)."""
    return Affine(res, 0, west, 0, -res, north)


def _shape_geometry(shape_type: str, pts: np.ndarray) -> BaseGeometry | None:
    """Shapely geometry (in the labelled tile's pixels) of one labelme shape."""
    if shape_type == "circle" and len(pts) >= 2:
        r = float(np.hypot(*(pts[1] - pts[0])))
        return Point(pts[0]).buffer(r, quad_segs=8)
    if len(pts) == 2:  # rectangle given by two corners
        return box(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    if len(pts) >= 3:  # polygon or oriented rectangle
        return Polygon(pts).buffer(0)
    return None


def tile_polygons(shapes, lab_transform: Affine, tile_tf: Affine, tile: int,
                  min_area_ratio: float) -> list[tuple[str, BaseGeometry]]:
    """Labelme shapes moved into a tile's pixels and clipped to it.

    The labelled tile and the 10 cm tile share the CRS and are north-up, so the change of pixels is a
    scale plus a shift. Returns (class, polygon in tile pixels), clipped to the tile.
    """
    m = ~tile_tf * lab_transform
    tile_square = box(0, 0, tile, tile)
    out = []
    for cls, shape_type, points in shapes:
        pts = np.asarray(points, dtype=np.float64)
        x = m.a * pts[:, 0] + m.b * pts[:, 1] + m.c
        y = m.d * pts[:, 0] + m.e * pts[:, 1] + m.f
        geom = _shape_geometry(shape_type, np.stack([x, y], axis=1))
        if geom is None or geom.is_empty or geom.area <= 0:
            continue
        clipped = geom.intersection(tile_square)
        if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if clipped.area < 4 or clipped.area / geom.area < min_area_ratio:
            continue
        out.append((cls, clipped))
    return out


def pixels_to_lonlat(geom: BaseGeometry, tile_tf: Affine, crs: str) -> BaseGeometry:
    """Tile-pixel geometry -> WGS84 geometry."""
    def fn(px, py):
        px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
        xs = tile_tf.a * px + tile_tf.b * py + tile_tf.c
        ys = tile_tf.d * px + tile_tf.e * py + tile_tf.f
        lon, lat = warp_transform(crs, LONLAT, xs.tolist(), ys.tolist())
        return lon, lat
    return shapely_transform(fn, geom)


def lonlat_to_pixels(geom: BaseGeometry, tile_tf: Affine, crs: str) -> BaseGeometry:
    """WGS84 geometry -> tile-pixel geometry (inverse of pixels_to_lonlat)."""
    inv = ~tile_tf

    def fn(lon, lat):
        xs, ys = warp_transform(LONLAT, crs, np.asarray(lon, dtype=float).tolist(),
                                np.asarray(lat, dtype=float).tolist())
        xs, ys = np.asarray(xs), np.asarray(ys)
        return inv.a * xs + inv.b * ys + inv.c, inv.d * xs + inv.e * ys + inv.f
    return shapely_transform(fn, geom)


def _polygons_of(geom: BaseGeometry) -> list[Polygon]:
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def export_tile_annotations(index_csv: str | Path = "data/tiles_10cm/tiles_index.csv",
                            labels_dir: str | Path = "data/labelme/images",
                            out_dir: str | Path = "data/tiles_10cm/labels",
                            tile: int = 640) -> Path:
    """Write one GeoJSON per annotated tile of the index into out_dir (the folder is rebuilt).

    Returns out_dir. Prints how many tiles were annotated and how many buildings each class has.
    """
    cfg = load_data_config()
    label_map, min_area_ratio = cfg["label_map"], cfg["min_area_ratio"]
    index = pd.read_csv(index_csv)
    footprints = labelled_footprints(labels_dir, label_map)
    if not footprints:
        raise FileNotFoundError(f"No labelled tiles (.tif + .json) in {labels_dir}")

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # generated folder: rebuilt from the labelme source every time
    out_dir.mkdir(parents=True)

    counts, n_annotated = Counter(), 0
    for row in index.itertuples(index=False):
        res = (row.east - row.west) / tile
        tf = tile_transform(row.west, row.north, res)
        bounds = (row.west, row.south, row.east, row.north)
        footprint = next((f for f in footprints if inside(bounds, f["bounds"])), None)
        if footprint is None:
            continue
        if footprint["crs"] != row.crs:
            raise ValueError(f"CRS mismatch: {footprint['name']} is {footprint['crs']}, tile is {row.crs}")

        features = []
        for cls, geom in tile_polygons(footprint["shapes"], footprint["transform"], tf, tile, min_area_ratio):
            lonlat = pixels_to_lonlat(geom, tf, row.crs)
            for poly in _polygons_of(lonlat):
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {"label": cls, "tile_id": row.tile_id, "source": footprint["name"]},
                })
                counts[cls] += 1
        doc = {"type": "FeatureCollection", "tile_id": row.tile_id, "scene_id": row.scene_id,
               "crs": LONLAT, "source": footprint["name"], "features": features}
        (out_dir / f"{row.tile_id}.geojson").write_text(json.dumps(doc), encoding="utf-8")
        n_annotated += 1

    print(f"{n_annotated} of {len(index)} tiles annotated -> {out_dir}/; buildings per class: {dict(counts)}")
    return out_dir


def read_tile_annotation(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
