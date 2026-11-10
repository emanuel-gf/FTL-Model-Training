"""Sliding-window inference over Cloud-Optimized GeoTIFFs, written as GeoJSON.

For each COG:
  1. the windows of `window` pixels are read directly with rasterio (no full-image load)
  2. the RT-DETR module predicts on the windows through PyTorch Lightning (Trainer.predict)
  3. detections are moved to full-image pixel coordinates and merged with class-wise NMS
  4. pixel boxes are turned into lon/lat polygons (via the COG transform and CRS) and
     written as one GeoJSON file per class
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

from src.dataset import to_model_input
from src.tiler import axis_starts, resample_to_cog


class WindowDataset(Dataset):
    """Windows of one COG. Returns (pixels, (x0, y0), valid fraction)."""

    def __init__(self, cog_path: Path, offsets: list[tuple[int, int]], window: int):
        self.cog_path = str(cog_path)
        self.offsets = offsets
        self.window = window

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int):
        x0, y0 = self.offsets[i]
        with rasterio.open(self.cog_path) as src:
            arr = src.read([1, 2, 3], window=Window(x0, y0, self.window, self.window), boundless=True,
                           fill_value=0)
        valid = float((arr.max(axis=0) > 0).mean())
        return to_model_input(np.moveaxis(arr, 0, -1)), torch.tensor([x0, y0], dtype=torch.float32), torch.tensor(valid)


def collate_windows(batch):
    images, offsets, valid = zip(*batch)
    return torch.stack(images), torch.stack(offsets), torch.stack(valid)


def window_offsets(width: int, height: int, window: int, stride: int) -> list[tuple[int, int]]:
    xs = axis_starts(width, window, stride)
    ys = axis_starts(height, window, stride)
    return [(x, y) for y in ys for x in xs]


def predict_cog(module: L.LightningModule, cog_path: Path, window: int, stride: int, batch_size: int,
                num_workers: int, min_valid: float, score_threshold: float, accelerator: str = "auto") -> dict:
    """Run the detector over one COG. Returns merged boxes (px), scores, labels and the window count."""
    with rasterio.open(cog_path) as src:
        width, height = src.width, src.height
    offsets = window_offsets(width, height, window, stride)

    module.min_valid = min_valid
    module.score_threshold = score_threshold
    module.window_scale = window / module.image_size

    loader = DataLoader(WindowDataset(cog_path, offsets, window), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_windows)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False, enable_progress_bar=True,
                        enable_model_summary=False)
    batches = trainer.predict(module, dataloaders=loader)

    boxes, scores, labels = [], [], []
    for batch_results in batches:
        for det in batch_results:
            boxes.append(det["boxes"])
            scores.append(det["scores"])
            labels.append(det["labels"])
    boxes = torch.cat(boxes) if boxes else torch.zeros((0, 4))
    scores = torch.cat(scores) if scores else torch.zeros((0,))
    labels = torch.cat(labels).long() if labels else torch.zeros((0,), dtype=torch.long)
    return {"boxes": boxes, "scores": scores, "labels": labels, "windows": len(offsets),
            "width": width, "height": height}


def merge_detections(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou: float):
    """Class-wise NMS: windows overlap, so the same building can be detected more than once."""
    if len(boxes) == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, iou)
    return boxes[keep], scores[keep], labels[keep]


def _pixel_polygon_to_lonlat(src, x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    xs, ys = zip(*(src.transform * (x, y) for x, y in corners))
    lon, lat = warp_transform(src.crs, "EPSG:4326", list(xs), list(ys))
    return [[round(a, 8), round(b, 8)] for a, b in zip(lon, lat)]


def detections_to_geojson(cog_path: Path, boxes, scores, labels, class_names: list[str],
                          out_dir: Path) -> list[Path]:
    """One GeoJSON per class: <out_dir>/<cog stem>_<class>.geojson (lon/lat, WGS84)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with rasterio.open(cog_path) as src:
        gsd = abs(src.transform.a)
        features_by_class: dict[str, list[dict]] = {name: [] for name in class_names}
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = box
            class_name = class_names[label]
            features_by_class[class_name].append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [_pixel_polygon_to_lonlat(src, x1, y1, x2, y2)]},
                "properties": {
                    "class": class_name,
                    "score": round(score, 4),
                    "scene": cog_path.stem,
                    "px_bbox": [round(v, 1) for v in box],
                    "area_m2": round((x2 - x1) * (y2 - y1) * gsd * gsd, 2),
                },
            })
    for class_name, feats in features_by_class.items():
        target = out_dir / f"{cog_path.stem}_{class_name}.geojson"
        target.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))
        written.append(target)
    return written


def prepare_cog(cog_path: Path, gsd: float | None, resampled_dir: Path) -> Path:
    """Use the COG as it is, or a copy resampled to `gsd` (written once in resampled_dir)."""
    if gsd is None:
        return cog_path
    dst = resampled_dir / cog_path.name
    if not dst.exists():
        print(f"resampling {cog_path.name} to {gsd * 100:g} cm")
        resample_to_cog(cog_path, dst, gsd)
    return dst
