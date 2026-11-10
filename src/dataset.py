"""Datasets and the PyTorch Lightning DataModule for tile detection.

Targets follow the Hugging Face RT-DETR convention: a dict per image with
'class_labels' (int64) and 'boxes' (float32, normalised cx, cy, w, h).
Splits are made by scene, so tiles of one scene never appear in two splits.
"""

from __future__ import annotations

import random
from pathlib import Path

import lightning as L
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.annotations import read_tile_samples
from src.labels import read_labelme_samples

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_input(img_hwc: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> normalised float tensor (3, H, W), ImageNet statistics."""
    x = torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def xyxy_to_cxcywh_norm(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([(x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                     (x2 - x1) / width, (y2 - y1) / height], axis=1).astype(np.float32)


def _augment(img: np.ndarray, boxes: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip, vertical flip and transpose, applied to image and boxes together."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:                                  # horizontal flip
        img = img[:, ::-1]
        if len(boxes):
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], axis=1)
    if rng.random() < 0.5:                                  # vertical flip
        img = img[::-1]
        if len(boxes):
            boxes = np.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], axis=1)
    if rng.random() < 0.5:                                  # transpose (swap x and y)
        img = img.transpose(1, 0, 2)
        if len(boxes):
            boxes = boxes[:, [1, 0, 3, 2]]
    return np.ascontiguousarray(img), boxes


class DetectionTileDataset(Dataset):
    """Tiles read with rasterio, resized to `image_size` if needed, with normalised targets."""

    def __init__(self, samples: list[dict], class_names: list[str], image_size: int, augment: bool = False,
                 seed: int = 0):
        self.samples = samples
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        with rasterio.open(s["tif"]) as src:
            img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        boxes = s["boxes"].astype(np.float32).copy()
        w, h = s["width"], s["height"]

        if (w, h) != (self.image_size, self.image_size):
            scale_x, scale_y = self.image_size / w, self.image_size / h
            img = np.asarray(Image.fromarray(img).resize((self.image_size, self.image_size), Image.BILINEAR))
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            w = h = self.image_size

        if self.augment:
            img, boxes = _augment(img, boxes, self.rng)

        labels = np.array([self.class_to_idx[c] for c in s["class_names"]], dtype=np.int64)
        target = {
            "class_labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(xyxy_to_cxcywh_norm(boxes, w, h)),
        }
        return to_model_input(img), target, s["tile_id"]


def collate_detection(batch):
    images, targets, names = zip(*batch)
    return torch.stack(images), list(targets), list(names)


class TileDataModule(L.LightningDataModule):
    """Splits the labelled tiles by scene and builds the train / val / test loaders."""

    def __init__(self, data_cfg: dict, train_cfg: dict, class_names: list[str], image_size: int):
        super().__init__()
        self.data_cfg = dict(data_cfg)
        self.train_cfg = dict(train_cfg)
        self.class_names = list(class_names)
        self.image_size = image_size
        self.split_table: list[dict] = []

    def setup(self, stage: str | None = None) -> None:
        cfg = self.data_cfg
        if "tiles_dir" in cfg:
            # the same tiles and GeoJSON annotations the notebooks use (see configs/data/buildings.yaml)
            samples = read_tile_samples(cfg["tiles_dir"], cfg.get("tile_size", 640))
            source = f"{cfg['tiles_dir']}/labels"
        else:
            samples = read_labelme_samples(cfg["labels_dir"], cfg["label_map"], cfg["min_area_ratio"],
                                           cfg.get("include_unlabelled", False))
            source = cfg["labels_dir"]
        if not samples:
            raise FileNotFoundError(f"No annotated tiles in {source}: run export_labels first")

        scenes = sorted({s["scene"] for s in samples})
        random.Random(cfg["split_seed"]).shuffle(scenes)
        n = len(scenes)
        n_test = round(n * cfg["test_fraction"])
        n_val = round(n * cfg["val_fraction"])
        test_scenes = set(scenes[:n_test])
        val_scenes = set(scenes[n_test:n_test + n_val])
        train_scenes = set(scenes[n_test + n_val:])
        if not train_scenes:
            raise ValueError(f"No training scenes left ({n} scenes, val={n_val}, test={n_test}). Lower the fractions.")

        def pick(keep: set[str]) -> list[dict]:
            return [s for s in samples if s["scene"] in keep]

        self.train_samples = pick(train_scenes)
        self.val_samples = pick(val_scenes)
        self.test_samples = pick(test_scenes)
        self.split_table = (
            [{"tile_id": s["tile_id"], "scene": s["scene"], "split": "train", "boxes": len(s["boxes"])} for s in self.train_samples]
            + [{"tile_id": s["tile_id"], "scene": s["scene"], "split": "val", "boxes": len(s["boxes"])} for s in self.val_samples]
            + [{"tile_id": s["tile_id"], "scene": s["scene"], "split": "test", "boxes": len(s["boxes"])} for s in self.test_samples]
        )

        seed = self.train_cfg.get("seed", 0)
        self.train_ds = DetectionTileDataset(self.train_samples, self.class_names, self.image_size,
                                             augment=self.train_cfg.get("augment", False), seed=seed)
        self.val_ds = DetectionTileDataset(self.val_samples, self.class_names, self.image_size)
        self.test_ds = DetectionTileDataset(self.test_samples, self.class_names, self.image_size)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.train_cfg["batch_size"], shuffle=shuffle,
                          num_workers=self.data_cfg["num_workers"], collate_fn=collate_detection)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        return self._loader(self.val_ds, shuffle=False) if len(self.val_ds) else None

    def test_dataloader(self) -> DataLoader | None:
        return self._loader(self.test_ds, shuffle=False) if len(self.test_ds) else None

    def save_split(self, path: str | Path) -> Path:
        import pandas as pd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.split_table).to_csv(path, index=False)
        return path
