"""FiftyOne dataset of building tiles: 640 x 640 px at 5 cm, with the labelme annotations as ground truth.

    dataset = build_dataset()           # tiles the COGs at 5 cm and adds every tile as a sample
    test = tag_test_samples(dataset)    # tags a few annotated tiles as "test" for the zero-shot check

Ground truth lives in the field `ground_truth` (fo.Detections, boxes in relative coordinates).
Only the classes of configs/data/buildings.yaml are kept: building and building-ext become
building, lote (parcels) is dropped.

A labelme annotation covers only the footprint of its labelled tile (data/labelme/images).
A tile gets labels only when it lies fully inside that footprint (annotated=True). Tiles outside
every footprint have no ground truth, so they are not used for evaluation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import fiftyone as fo
import numpy as np
import pandas as pd
import rasterio
import yaml
from fiftyone import ViewField as F
from rasterio.transform import Affine

from src.labels import shape_to_box
from src.tiler import tile_cogs

DATASET_NAME = "building_tiles_5cm"
DATA_CONFIG = Path("configs/data/buildings.yaml")


def _labelled_footprints(labels_dir: str | Path, label_map: dict) -> list[dict]:
    """One entry per labelled tile: bounds, transform, CRS and the kept shapes (in its pixels)."""
    footprints = []
    for tif in sorted(Path(labels_dir).glob("*.tif")):
        json_path = tif.with_suffix(".json")
        if not json_path.exists():
            continue
        with rasterio.open(tif) as src:
            footprints.append({
                "name": tif.name,
                "bounds": tuple(src.bounds),
                "transform": src.transform,
                "crs": src.crs.to_string(),
                "shapes": _kept_shapes(json_path, label_map),
            })
    return footprints


def _kept_shapes(json_path: Path, label_map: dict) -> list[tuple[str, str, list]]:
    """(class, shape_type, points) for every labelme shape whose label maps to a class."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = []
    for shape in data.get("shapes", []):
        cls = label_map.get(shape.get("label"))
        if cls is not None and shape.get("points"):
            shapes.append((cls, shape.get("shape_type", "polygon"), shape["points"]))
    return shapes


def _inside(inner: tuple, outer: tuple, tol: float = 1e-3) -> bool:
    """True if the bounds (west, south, east, north) `inner` lie inside `outer`."""
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _tile_labels(shapes, lab_transform: Affine, tile_transform: Affine, tile: int,
                 min_area_ratio: float) -> list[tuple[str, list[float]]]:
    """Move labelme shapes from the labelled tile's pixels into this tile's pixels.

    Both rasters share the CRS and are north-up, so the mapping is a pure scale + shift. Each box is
    clipped to the tile and kept only if at least `min_area_ratio` of it stays inside (as in src/labels.py).
    Returns (class, [x, y, w, h]) with the box relative to the tile.
    """
    m = ~tile_transform * lab_transform
    out = []
    for cls, shape_type, points in shapes:
        pts = np.asarray(points, dtype=np.float64)
        x = m.a * pts[:, 0] + m.b * pts[:, 1] + m.c
        y = m.d * pts[:, 0] + m.e * pts[:, 1] + m.f
        box = shape_to_box(shape_type, np.stack([x, y], axis=1).tolist())
        if box is None:
            continue
        x1, y1, x2, y2 = box
        full_area = (x2 - x1) * (y2 - y1)
        if full_area <= 0:
            continue
        cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
        cx2, cy2 = min(x2, tile), min(y2, tile)
        w, h = cx2 - cx1, cy2 - cy1
        if w < 2 or h < 2 or (w * h) / full_area < min_area_ratio:
            continue
        out.append((cls, [cx1 / tile, cy1 / tile, w / tile, h / tile]))
    return out


def build_dataset(cog_dir: str = "cog", labels_dir: str = "data/labelme/images",
                  tiles_dir: str = "data/tiles_5cm", gsd: float = 0.05, tile: int = 640,
                  overlap: int = 128, min_valid: float = 0.9, retile: bool = True,
                  name: str = DATASET_NAME, overwrite: bool = False) -> fo.Dataset:
    """Tile every COG at `gsd` m/px and build the FiftyOne dataset with the labelme ground truth.

    retile=True re-tiles the COGs (needed the first time or after a change of gsd/tile/overlap).
    If the dataset already exists it is loaded, unless overwrite=True rebuilds it.
    """
    if fo.dataset_exists(name):
        if not overwrite:
            print(f"dataset '{name}' already exists: loading it (overwrite=True rebuilds it)")
            return fo.load_dataset(name)
        fo.delete_dataset(name)

    cfg = yaml.safe_load(DATA_CONFIG.read_text())
    label_map, min_area_ratio = cfg["label_map"], cfg["min_area_ratio"]

    tiles_dir = Path(tiles_dir)
    index_csv = tiles_dir / "tiles_index.csv"
    if retile or not index_csv.exists():
        # a separate resampled folder per gsd: tile_cogs reuses resampled files by name
        resampled = Path(cog_dir) / f"resampled_{round(gsd * 100)}cm"
        tile_cogs(cog_dir, tiles_dir, gsd=gsd, tile=tile, overlap=overlap,
                  min_valid=min_valid, resampled_dir=resampled)
    index = pd.read_csv(index_csv)

    footprints = _labelled_footprints(labels_dir, label_map)
    samples, counts, n_annotated = [], Counter(), 0
    for row in index.itertuples(index=False):
        res = (row.east - row.west) / tile
        tile_transform = Affine(res, 0, row.west, 0, -res, row.north)
        bounds = (row.west, row.south, row.east, row.north)

        footprint = next((f for f in footprints if _inside(bounds, f["bounds"])), None)
        detections = []
        if footprint is not None:
            if footprint["crs"] != row.crs:
                raise ValueError(f"CRS mismatch: {footprint['name']} is {footprint['crs']}, tile is {row.crs}")
            for cls, box in _tile_labels(footprint["shapes"], footprint["transform"],
                                         tile_transform, tile, min_area_ratio):
                detections.append(fo.Detection(label=cls, bounding_box=[float(v) for v in box]))
            counts.update(d.label for d in detections)
            n_annotated += 1

        samples.append(fo.Sample(
            filepath=str(Path(row.tile_file).resolve()),
            tile_id=row.tile_id,
            scene_id=row.scene_id,
            gsd_m=float(res),
            valid_fraction=float(row.valid_fraction),
            centre_lat=float(row.centre_lat),
            centre_lon=float(row.centre_lon),
            annotated=footprint is not None,
            label_source=footprint["name"] if footprint is not None else None,
            ground_truth=fo.Detections(detections=detections),
        ))

    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(samples)
    print(f"{len(samples)} tiles from {index['scene_id'].nunique()} scenes, "
          f"{n_annotated} annotated; boxes per class: {dict(counts)}")
    return dataset


def tag_test_samples(dataset: fo.Dataset, n: int = 10, seed: int = 51, tag: str = "test") -> fo.DatasetView:
    """Remove the old `tag` and tag `n` random annotated tiles with it. Returns the tagged view."""
    dataset.untag_samples(tag)
    view = dataset.match(F("annotated") == True).shuffle(seed=seed).limit(n)  # noqa: E712
    view.tag_samples(tag)
    scenes = len(set(view.values("scene_id")))
    print(f"tagged {view.count()} annotated tiles as '{tag}' from {scenes} scene(s)")
    return dataset.match_tags(tag)


def dataset_summary(dataset: fo.Dataset) -> None:
    print(f"{dataset.name}: {len(dataset)} samples")
    print("annotated:", dataset.count_values("annotated"))
    print("ground truth labels:", dataset.count_values("ground_truth.detections.label"))
