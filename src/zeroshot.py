"""Pretrained building detectors (RT-DETR, OWLv2, SAM2). Runs under uv, writes files only.

Two kinds of output, read by the notebook kernel (FiftyOne lives there, not here):
- predict_tiles: one JSON per annotated tile in outputs/predictions/<name>/<tile_id>.json
  (pixel boxes of the tile, so the kernel can attach them to the FiftyOne samples)
- predict_scenes: one GeoJSON per scene in outputs/batch/<name>/<scene_id>.geojson (WGS84)

Every prediction is labelled "building". The original model label is kept in `source_label`:
- rtdetr: COCO-trained RT-DETR has no building class, so every COCO detection counts as a building.
  This is a baseline, not a building detector. A fine-tuned checkpoint with a `building` class works too.
- owlv2: open-vocabulary, the text queries are the prompt (default "a building").
- sam2: automatic mask generation is class-agnostic. Each mask is kept by score and footprint area only.

Use the CLI run_models.py to call these from the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (AutoImageProcessor, Owlv2ForObjectDetection, Owlv2Processor,
                          RTDetrForObjectDetection, pipeline)

from src.annotations import tile_transform


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RTDetrDetector:
    """RT-DETR from the hub or a local folder (COCO weights or a fine-tuned checkpoint)."""

    def __init__(self, weights: str, device: str, score_threshold: float = 0.3, **_):
        self.device = device
        self.threshold = score_threshold
        self.processor = AutoImageProcessor.from_pretrained(weights)
        self.model = RTDetrForObjectDetection.from_pretrained(weights).to(device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, gsd_m: float) -> list[dict]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(image.height, image.width)])[0]
        names = self.model.config.id2label
        return [{"label": "building", "source_label": names[int(lab)], "score": float(s), "box": b}
                for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist())]


class Owlv2Detector:
    """OWLv2 open-vocabulary detector. Each text query is a prompt, all of them count as buildings."""

    def __init__(self, weights: str, device: str, score_threshold: float = 0.2,
                 queries: tuple[str, ...] | list[str] = ("a building",), **_):
        self.device = device
        self.threshold = score_threshold
        self.queries = list(queries)
        self.processor = Owlv2Processor.from_pretrained(weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(weights).to(device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, gsd_m: float) -> list[dict]:
        inputs = self.processor(text=[self.queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target = torch.tensor([[image.height, image.width]], device=self.device)
        # the processor does not have this method in every transformers version; the image processor does
        res = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target)[0]
        return [{"label": "building", "source_label": self.queries[int(lab)], "score": float(s), "box": b}
                for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist())]


class Sam2MaskDetector:
    """SAM2 automatic mask generation. Masks become building candidates, kept by score and area in m2."""

    def __init__(self, weights: str, device: str, score_threshold: float = 0.85,
                 min_area_m2: float = 10.0, max_area_m2: float = 1500.0,
                 points_per_batch: int = 64, **_):
        self.threshold = score_threshold
        self.min_area = min_area_m2
        self.max_area = max_area_m2
        self.points_per_batch = points_per_batch
        self.generator = pipeline("mask-generation", model=weights, device=device)

    def predict(self, image: Image.Image, gsd_m: float) -> list[dict]:
        out = self.generator(image, points_per_batch=self.points_per_batch)
        dets = []
        for mask, score in zip(out["masks"], out["scores"]):
            m = np.asarray(mask) > 0
            ys, xs = np.nonzero(m)
            if xs.size == 0 or float(score) < self.threshold:
                continue
            area_m2 = float(m.sum()) * gsd_m ** 2
            if not self.min_area <= area_m2 <= self.max_area:
                continue
            dets.append({"label": "building", "source_label": "sam2_mask", "score": float(score),
                         "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]})
        return dets


DETECTORS = {"rtdetr": RTDetrDetector, "owlv2": Owlv2Detector, "sam2": Sam2MaskDetector}


def load_detector(kind: str, weights: str, device: str, **options):
    """kind: 'rtdetr' | 'owlv2' | 'sam2'. `options` are the detector's keyword arguments."""
    if kind not in DETECTORS:
        raise ValueError(f"unknown kind {kind!r}, choose from {sorted(DETECTORS)}")
    return DETECTORS[kind](weights, device, **options)


def predict_tiles(detector, index_csv: str, labels_dir: str, out_dir: str, tile: int = 640) -> int:
    """Run the detector on every annotated tile (a GeoJSON exists in labels_dir) and write one JSON each.

    JSON: {"tile_id", "width", "height", "detections": [{"label", "source_label", "score", "box": [x1, y1, x2, y2]}]}
    with the box in tile pixels. Returns the number of tiles written.
    """
    index = pd.read_csv(index_csv)
    labels_dir = Path(labels_dir)
    rows = [r for r in index.itertuples(index=False) if (labels_dir / f"{r.tile_id}.geojson").exists()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image = Image.open(row.tile_file).convert("RGB")
        res = (row.east - row.west) / tile
        dets = []
        for p in detector.predict(image, res):
            x1, y1 = max(p["box"][0], 0.0), max(p["box"][1], 0.0)
            x2, y2 = min(p["box"][2], image.width), min(p["box"][3], image.height)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({"label": p["label"], "source_label": p["source_label"], "score": p["score"],
                         "box": [x1, y1, x2, y2]})
        doc = {"tile_id": row.tile_id, "width": image.width, "height": image.height, "detections": dets}
        (out / f"{row.tile_id}.json").write_text(json.dumps(doc), encoding="utf-8")
    return len(rows)


def predict_scenes(detector, index_csv: str, out_dir: str, scenes: list[str] | None = None,
                   nms_iou: float = 0.5, tile: int = 640) -> dict[str, int]:
    """Run the detector over every tile of the chosen scenes and write one GeoJSON per scene.

    Each box is moved from tile pixels to the CRS of the tiles (metres). Boxes of overlapping tiles are
    merged with NMS, and the result is written as WGS84 polygons to <out_dir>/<scene_id>.geojson.
    scenes=None runs every scene of the index. Returns {scene_id: number of buildings written}.
    """
    from shapely.geometry import Polygon, mapping
    from torchvision.ops import nms

    index = pd.read_csv(index_csv)
    if scenes:
        index = index[index["scene_id"].isin(scenes)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    for scene_id, rows in index.groupby("scene_id", sort=True):
        boxes, scores, owners, crs = [], [], [], None
        for row in rows.itertuples(index=False):
            crs = row.crs
            res = (row.east - row.west) / tile
            tf = tile_transform(row.west, row.north, res)
            image = Image.open(row.tile_file).convert("RGB")
            for p in detector.predict(image, res):
                x1, y1 = max(p["box"][0], 0.0), max(p["box"][1], 0.0)
                x2, y2 = min(p["box"][2], image.width), min(p["box"][3], image.height)
                if x2 <= x1 or y2 <= y1:
                    continue
                # tile pixels -> CRS metres: north-up, so the row index runs down from the top edge
                X1, Y1 = tf.c + tf.a * x1, tf.f + tf.e * y1
                X2, Y2 = tf.c + tf.a * x2, tf.f + tf.e * y2
                boxes.append([min(X1, X2), min(Y1, Y2), max(X1, X2), max(Y1, Y2)])
                scores.append(p["score"])
                owners.append(row.tile_id)

        features = []
        if boxes:
            keep = nms(torch.tensor(boxes, dtype=torch.float64),
                       torch.tensor(scores, dtype=torch.float64), nms_iou).tolist()
            for k in keep:
                x1, y1, x2, y2 = boxes[k]
                poly = _metres_to_lonlat(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), crs)
                features.append({"type": "Feature", "geometry": mapping(poly),
                                 "properties": {"label": "building", "score": scores[k],
                                                "tile_id": owners[k]}})
            print(f"{scene_id}: {len(boxes)} boxes -> {len(features)} buildings after NMS")
        doc = {"type": "FeatureCollection", "scene_id": scene_id, "crs": "EPSG:4326", "features": features}
        (out / f"{scene_id}.geojson").write_text(json.dumps(doc), encoding="utf-8")
        written[scene_id] = len(features)
    return written


def _metres_to_lonlat(geom, crs: str):
    from rasterio.warp import transform as warp_transform
    from shapely.ops import transform as shapely_transform

    def fn(x, y):
        return warp_transform(crs, "EPSG:4326", list(x), list(y))
    return shapely_transform(fn, geom)
