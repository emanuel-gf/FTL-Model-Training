"""Shared start of the two notebooks (stdlib only). Both call setup(), only `group` differs.

    model_train.ipynb        setup("gpu", ...)   cloud: torch, transformers, lightning
    fiftyone_workflow.ipynb  setup("fo", ...)    your computer: FiftyOne viewer, no torch
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def in_cloud() -> bool:
    import os
    return "COLAB_RELEASE_TAG" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def setup(group: str, tiles_dir: str) -> Path:
    """Sync the uv environment for `group`, make sure uv exists, and report the data paths."""
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["uv", "sync", "--group", group], check=True)

    tiles = Path(tiles_dir)
    index = tiles / "tiles_index.csv"
    labels = tiles / "labels"
    print(f"environment: uv group '{group}'")
    print(f"tiles folder:  {tiles}   ({'found' if index.exists() else 'NOT FOUND: run the tiling step'})")
    print(f"labels folder: {labels}   ({len(list(labels.glob('*.geojson'))) if labels.exists() else 0} annotated tiles)")
    return tiles
