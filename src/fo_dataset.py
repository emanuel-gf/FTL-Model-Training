"""FiftyOne side of the pipeline. Runs in the notebook kernel (pip), never under uv.

uv produces the files, this module only reads them:
    tile_cogs.py          -> data/tiles_10cm/tiles/*.tif + tiles_index.csv
    export_annotations.py -> data/tiles_10cm/labels/<tile_id>.geojson       (ground truth)
    run_models.py         -> outputs/predictions/<name>/<tile_id>.json      (per-tile predictions)

    dataset = build_dataset()                 # one sample per tile, ground truth from the GeoJSON
    test = tag_test_samples(dataset, n=10)    # a few annotated tiles tagged "test"
    add_predictions(test, field, pred_dir)    # attach a model's predictions
    summary = evaluate_run(test, field)       # IoU 0.5 metrics against ground_truth

Boxes are stored in FiftyOne's relative format [x, y, w, h] in [0, 1].
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fiftyone as fo
import pandas as pd
from fiftyone import ViewField as F
from shapely.geometry import shape

from src.annotations import lonlat_to_pixels, tile_transform

DATASET_NAME = "building_tiles_10cm"


def tile_boxes_from_annotation(path: Path, tile_tf, crs: str, tile: int) -> list[tuple[str, list[float]]]:
    """(label, relative [x, y, w, h]) for the buildings in one tile's GeoJSON (bounds of each polygon)."""
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


def build_dataset(tiles_dir: str = "data/tiles_10cm", labels_dir: str = "data/tiles_10cm/labels",
                  tile: int = 640, name: str = DATASET_NAME, overwrite: bool = False) -> fo.Dataset:
    """One FiftyOne sample per tile of tiles_index.csv, with the GeoJSON annotations as `ground_truth`.

    The tiles must already exist (tile_cogs.py) and the annotations must be exported (export_annotations.py).
    If the dataset already exists it is loaded, unless overwrite=True rebuilds it.
    """
    if fo.dataset_exists(name):
        if not overwrite:
            print(f"dataset '{name}' already exists: loading it (overwrite=True rebuilds it)")
            return fo.load_dataset(name)
        fo.delete_dataset(name)

    index = pd.read_csv(Path(tiles_dir) / "tiles_index.csv")
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


def run_name(kind: str, weights: str) -> str:
    """Name of a model run: the prediction folder and the FiftyOne field, e.g. 'owlv2_owlv2_base_patch16_ensemble'."""
    base = weights.rstrip("/").split("/")[-1]
    return re.sub(r"\W+", "_", f"{kind}_{base}").strip("_").lower()


def add_predictions(view: fo.DatasetView, field: str, pred_dir: str) -> int:
    """Attach the per-tile prediction JSON files of `pred_dir` to `field` of the samples in `view`.

    Samples without a file get an empty field. Returns how many samples had predictions.
    """
    pred_dir = Path(pred_dir)
    found = 0
    for sample in view.iter_samples(progress=True, autosave=True):
        path = pred_dir / f"{sample.tile_id}.json"
        dets = []
        if path.exists():
            found += 1
            doc = json.loads(path.read_text(encoding="utf-8"))
            w, h = doc["width"], doc["height"]
            for p in doc["detections"]:
                x1, y1, x2, y2 = p["box"]
                dets.append(fo.Detection(
                    label=p["label"], confidence=p["score"], source_label=p["source_label"],
                    bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]))
        sample[field] = fo.Detections(detections=dets)
    print(f"{found} of {len(view)} tiles had predictions in {pred_dir}/")
    return found


def evaluate_run(view: fo.DatasetView, field: str, iou: float = 0.5) -> dict:
    """Score `field` against `ground_truth` on `view`. Returns mAP, precision, recall, F1 and counts."""
    eval_key = f"eval_{field}"
    results = view.evaluate_detections(field, gt_field="ground_truth", eval_key=eval_key,
                                       iou=iou, compute_mAP=True)
    metrics = results.metrics()
    return {
        "mAP_50_95": results.mAP(),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["fscore"],
        "tp": view.sum(f"{eval_key}_tp"),
        "fp": view.sum(f"{eval_key}_fp"),
        "fn": view.sum(f"{eval_key}_fn"),
        "predictions": view.count_values(f"{field}.detections.label").get("building", 0),
    }


def compare_runs(summary: dict[str, dict]) -> pd.DataFrame:
    """Table with one row per run (the output of evaluate_run, keyed by field name)."""
    df = pd.DataFrame(summary).T
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    return df


def dataset_summary(dataset: fo.Dataset) -> None:
    print(f"{dataset.name}: {len(dataset)} samples")
    print("annotated:", dataset.count_values("annotated"))
    print("ground truth labels:", dataset.count_values("ground_truth.detections.label"))
