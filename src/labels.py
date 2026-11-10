"""labelme annotations -> boxes for the detector.

Each tile is a GeoTIFF and its labelme annotation is a .json with the same name.
A polygon, rectangle, oriented rectangle or circle becomes an axis-aligned box in
pixels; it is clipped to the tile, and kept only if at least `min_area_ratio` of
its area stays inside the tile.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import rasterio

BOX_SHAPES = {"polygon", "rectangle", "oriented_rectangle", "circle"}


def tile_scene_id(stem: str) -> str:
    """Scene of a tile: '<item_id>_<n>' -> '<item_id>' (tiles from one scene share the item ID)."""
    return re.sub(r"_\d+$", "", stem)


def shape_to_box(shape_type: str, points: list[list[float]]) -> np.ndarray | None:
    """Axis-aligned box (x1, y1, x2, y2) of a labelme shape, or None if the shape is not a box."""
    pts = np.asarray(points, dtype=np.float64)
    if shape_type == "circle" and len(pts) >= 2:
        centre, edge = pts[0], pts[1]
        r = float(np.hypot(*(edge - centre)))
        return np.array([centre[0] - r, centre[1] - r, centre[0] + r, centre[1] + r])
    if shape_type in BOX_SHAPES and len(pts) >= 2:
        return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return None


def load_boxes(json_path: Path, width: int, height: int, label_map: dict,
               min_area_ratio: float) -> tuple[np.ndarray, list[str]]:
    """Boxes (N, 4) in pixels and their class names for one annotated tile."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    boxes, classes = [], []
    for shape in data.get("shapes", []):
        class_name = label_map.get(shape.get("label"))
        if class_name is None:
            continue
        box = shape_to_box(shape.get("shape_type", "polygon"), shape.get("points", []))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        full_area = (x2 - x1) * (y2 - y1)
        clipped = np.array([max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)])
        w, h = clipped[2] - clipped[0], clipped[3] - clipped[1]
        if w < 2 or h < 2 or full_area <= 0:
            continue
        if (w * h) / full_area < min_area_ratio:
            continue
        boxes.append(clipped)
        classes.append(class_name)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), []
    return np.asarray(boxes, dtype=np.float32), classes


def read_labelme_samples(labels_dir: str | Path, label_map: dict, min_area_ratio: float,
                         include_unlabelled: bool = False) -> list[dict]:
    """One sample per tile: path, scene, size and boxes. Tiles without .json are skipped unless asked."""
    samples = []
    for tif in sorted(Path(labels_dir).glob("*.tif")):
        json_path = tif.with_suffix(".json")
        if not json_path.exists() and not include_unlabelled:
            continue
        with rasterio.open(tif) as src:
            width, height = src.width, src.height
        if json_path.exists():
            boxes, classes = load_boxes(json_path, width, height, label_map, min_area_ratio)
        else:
            boxes, classes = np.zeros((0, 4), dtype=np.float32), []
        samples.append({
            "tif": str(tif),
            "tile_id": tif.stem,
            "scene": tile_scene_id(tif.stem),
            "width": width,
            "height": height,
            "boxes": boxes,
            "class_names": classes,
        })
    return samples
