"""Embedding files: one .npz per model and resolution, with the tile ids stored inside it.

Written by embed.py (cloud, GPU), read locally by fiftyone_workflow.ipynb. Numpy only, no torch.

Arrays in the file:
  vectors       float32 (N, D)  one row per tile, L2-normalised
  tile_id       str (N,)        links each row to a tile and to its FiftyOne sample
  scene_id      str (N,)
  bounds        float64 (N, 4)  west, south, east, north of each tile, to check the match
Scalars in the file (0-d arrays):
  crs, resolution_m, tile_size, model

Why the ids are inside the file: the vectors and their ids are saved together, so they can never
get out of order, and one file can be copied or downloaded on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_embeddings(path: str | Path, vectors: np.ndarray, tile_ids, scene_ids, bounds: np.ndarray,
                    crs: str, resolution_m: float, tile_size: int, model: str, input_px: int = 0) -> Path:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    n = len(vectors)
    if not (len(tile_ids) == len(scene_ids) == len(bounds) == n):
        raise ValueError("vectors, tile_ids, scene_ids and bounds must have the same length")
    if len(set(tile_ids)) != n:
        raise ValueError("tile_ids must be unique")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors,
        tile_id=np.asarray(tile_ids, dtype=str),
        scene_id=np.asarray(scene_ids, dtype=str),
        bounds=np.asarray(bounds, dtype=np.float64),
        crs=np.array(crs), resolution_m=np.array(float(resolution_m)),
        tile_size=np.array(int(tile_size)), model=np.array(model), input_px=np.array(int(input_px)),
    )
    return path


def load_embeddings(path: str | Path) -> dict:
    """Read an embedding file. Returns the arrays and the scalars as plain Python values."""
    with np.load(Path(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return {
        "vectors": data["vectors"].astype(np.float32),
        "tile_id": [str(t) for t in data["tile_id"]],
        "scene_id": [str(s) for s in data["scene_id"]],
        "bounds": data["bounds"],
        "crs": str(data["crs"]),
        "resolution_m": float(data["resolution_m"]),
        "tile_size": int(data["tile_size"]),
        "model": str(data["model"]),
        "input_px": int(data["input_px"]) if "input_px" in data else 0,
    }


def row_of(emb: dict) -> dict[str, int]:
    """tile_id -> row number in the vectors."""
    return {t: i for i, t in enumerate(emb["tile_id"])}
