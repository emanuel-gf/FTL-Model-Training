"""FiftyOne dataset of building tiles: 640 x 640 px at 10 cm, with the building annotations as ground truth.

    export_tile_annotations()          # labelme -> data/tiles_10cm/labels/<tile_id>.geojson
    dataset = build_dataset()          # tiles the COGs at 10 cm and adds every tile as a sample
    test = tag_test_samples(dataset)   # tags a few annotated tiles as "test" for the zero-shot check

Ground truth is the field `ground_truth` (fo.Detections, boxes in relative coordinates), read from
the per-tile GeoJSON in data/tiles_10cm/labels. A tile is annotated when its GeoJSON exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import fiftyone as fo
import pandas as pd
from fiftyone import ViewField as F
from shapely.geometry import shape

from src.annotations import lonlat_to_pixels, tile_transform
from src.tiler import tile_cogs

DATASET_NAME = "building_tiles_10cm"


def tile_boxes_from_annotation(path: Path, tile_tf, crs: str, tile: int) -> list[list[float]]:
    """Relative [x, y, w, h] boxes of the buildings in one tile's GeoJSON (bounds of each polygon)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for feat in doc.get("features", []):
        geom = lonlat_to_pixels(shape(feat["geometry"]), tile_tf, crs)
        x1, y1, x2, y2 = geom.bounds
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, tile), min(y2, tile)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((feat["properties"]["label"],
                      [x1 / tile, y1 / tile, (x2 - x1) / tile, (y2 - y1) / tile]))
    return boxes


def build_dataset(cog_dir: str = "cog", labels_dir: str = "data/tiles_10cm/labels",
                  tiles_dir: str = "data/tiles_10cm", gsd: float = 0.10, tile: int = 640,
                  overlap: int = 128, min_valid: float = 0.9, retile: bool = True,
                  name: str = DATASET_NAME, overwrite: bool = False) -> fo.Dataset:
    """Tile every COG at `gsd` m/px and build the FiftyOne dataset with the GeoJSON annotations.

    retile=True re-tiles the COGs. If the dataset already exists it is loaded, unless overwrite=True.
    Run export_tile_annotations() first, so that labels_dir is up to date.
    """
    if fo.dataset_exists(name):
        if not overwrite:
            print(f"dataset '{name}' already exists: loading it (overwrite=True rebuilds it)")
            return fo.load_dataset(name)
        fo.delete_dataset(name)

    tiles_dir = Path(tiles_dir)
    index_csv = tiles_dir / "tiles_index.csv"
    if retile or not index_csv.exists():
        # a separate resampled folder per gsd: tile_cogs reuses resampled files by name
        resampled = Path(cog_dir) / f"resampled_{round(gsd * 100)}cm"
        tile_cogs(cog_dir, tiles_dir, gsd=gsd, tile=tile, overlap=overlap,
                  min_valid=min_valid, resampled_dir=resampled)
    index = pd.read_csv(index_csv)
    labels_dir = Path(labels_dir)

    samples, n_annotated, n_boxes = [], 0, {}
    for row in index.itertuples(index=False):
        res = (row.east - row.west) / tile
        tf = tile_transform(row.west, row.north, res)
        ann_path = labels_dir / f"{row.tile_id}.geojson"
        annotated = ann_path.exists()
        detections = []
        if annotated:
            n_annotated += 1
            for cls, box in tile_boxes_from_annotation(ann_path, tf, row.crs, tile):
                detections.append(fo.Detection(label=cls, bounding_box=[float(v) for v in box]))
                n_boxes[cls] = n_boxes.get(cls, 0) + 1

        samples.append(fo.Sample(
            filepath=str(Path(row.tile_file).resolve()),
            tile_id=row.tile_id,
            scene_id=row.scene_id,
            gsd_m=float(res),
            valid_fraction=float(row.valid_fraction),
            centre_lat=float(row.centre_lat),
            centre_lon=float(row.centre_lon),
            annotated=annotated,
            ground_truth=fo.Detections(detections=detections),
        ))

    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(samples)
    print(f"{len(samples)} tiles from {index['scene_id'].nunique()} scenes, "
          f"{n_annotated} annotated; boxes per class: {n_boxes}")
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
