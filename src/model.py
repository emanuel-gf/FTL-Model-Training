"""RT-DETR as a PyTorch Lightning module: fine-tuning, validation, test and prediction.

- build_rtdetr: load any RT-DETR checkpoint from the Hugging Face hub or a local folder,
  with a new classification head for our classes.
- apply_freezing: freeze the backbone, the encoder or the first decoder layers.
- RTDetrLightning: training_step, validation/test (loss + COCO mAP), predict_step.
- download_model: save a hub checkpoint into a local folder (for offline use or Kaggle).
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoImageProcessor, RTDetrForObjectDetection


def build_rtdetr(source: str | Path, class_names: list[str]) -> RTDetrForObjectDetection:
    """RT-DETR from a hub id or a local folder. The head is re-initialised for `class_names`."""
    id2label = dict(enumerate(class_names))
    label2id = {name: i for i, name in id2label.items()}
    return RTDetrForObjectDetection.from_pretrained(
        str(source), num_labels=len(class_names), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _freeze(module: nn.Module) -> int:
    n = 0
    for p in module.parameters():
        p.requires_grad_(False)
        n += p.numel()
    return n


def apply_freezing(model: RTDetrForObjectDetection, train_cfg: dict) -> dict:
    """Freeze the parts requested in the config. Returns the number of frozen parameters."""
    core = model.model
    frozen = {"backbone": 0, "encoder": 0, "decoder_layers": 0}
    if train_cfg.get("freeze_backbone"):
        frozen["backbone"] = _freeze(core.backbone)
    if train_cfg.get("freeze_encoder"):
        frozen["encoder"] = _freeze(core.encoder)
    for layer in core.decoder.layers[: int(train_cfg.get("freeze_decoder_layers", 0))]:
        frozen["decoder_layers"] += _freeze(layer)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters: {frozen} | trainable {trainable / 1e6:.1f} M of {total / 1e6:.1f} M")
    return frozen


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def decode(logits: torch.Tensor, pred_boxes: torch.Tensor, size: float, threshold: float,
           topk: int = 300) -> list[dict]:
    """RT-DETR outputs -> per-image detections (xyxy in pixels of the model input).

    As in RT-DETR, scores are sigmoid(logits) and the top-k (query, class) pairs are kept
    over all queries and classes at once.
    """
    scores = logits.sigmoid()
    batch, _, n_classes = scores.shape
    results = []
    for b in range(batch):
        flat = scores[b].reshape(-1)
        top, idx = flat.topk(min(topk, flat.numel()))
        query = torch.div(idx, n_classes, rounding_mode="floor")
        label = idx % n_classes
        keep = top >= threshold
        boxes = cxcywh_to_xyxy(pred_boxes[b][query[keep]]) * float(size)
        results.append({"boxes": boxes, "scores": top[keep], "labels": label[keep]})
    return results


class RTDetrLightning(L.LightningModule):
    """RT-DETR for tiles of `image_size` pixels. `train_cfg` is only needed for training."""

    def __init__(self, model_source: str | Path, class_names: list[str], train_cfg: dict | None = None,
                 image_size: int = 640, score_threshold: float = 0.4):
        super().__init__()
        self.class_names = list(class_names)
        self.image_size = image_size
        self.train_cfg = dict(train_cfg or {})
        self.score_threshold = score_threshold
        self.min_valid = 0.0          # predict_step: skip windows with less data than this
        self.window_scale = 1.0       # predict_step: window size / model input size

        self.model = build_rtdetr(model_source, self.class_names)
        if train_cfg:
            apply_freezing(self.model, self.train_cfg)
        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                               backend="faster_coco_eval")

    def forward(self, pixel_values: torch.Tensor, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    # ── training ─────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        out = self(images, targets)
        self.log("train/loss", out.loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=images.shape[0])
        return out.loss

    def configure_optimizers(self):
        cfg = self.train_cfg
        backbone, others = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if name.startswith("model.backbone") else others).append(p)
        lr = float(cfg.get("lr", 1e-4))
        groups = [{"params": others, "lr": lr}]
        if backbone:
            groups.append({"params": backbone, "lr": lr * float(cfg.get("backbone_lr_mult", 1.0))})
        optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.get("weight_decay", 1e-4)))
        if cfg.get("scheduler", "none") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 1)))
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    # ── validation and test: loss and COCO mAP ─────────────────────────────────
    def _evaluate(self, batch, stage: str) -> None:
        images, targets, _ = batch
        out = self(images, targets)
        self.log(f"{stage}/loss", out.loss, on_epoch=True, batch_size=images.shape[0])
        preds = decode(out.logits, out.pred_boxes, self.image_size, threshold=0.001)
        gts = []
        for t in targets:
            gts.append({
                "boxes": cxcywh_to_xyxy(t["boxes"].to(images.device)) * float(self.image_size),
                "labels": t["class_labels"].to(images.device),
            })
        self.map_metric.update(preds, gts)

    def _finish_evaluation(self, stage: str) -> None:
        result = self.map_metric.compute()
        self.log(f"{stage}/map_50", result["map_50"].float(), prog_bar=True)
        self.log(f"{stage}/map", result["map"].float())
        self.map_metric.reset()

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def on_validation_epoch_end(self):
        self._finish_evaluation("val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def on_test_epoch_end(self):
        self._finish_evaluation("test")

    # ── prediction over windows ───────────────────────────────────────────────
    def predict_step(self, batch, batch_idx):
        """batch = (pixel_values, offsets (x0, y0), valid fraction). Returns boxes in full-image pixels."""
        images, offsets, valid = batch
        out = self.model(pixel_values=images)
        detections = decode(out.logits, out.pred_boxes, self.image_size, self.score_threshold)
        results = []
        for det, offset, v in zip(detections, offsets, valid):
            if float(v) < self.min_valid:
                det = {"boxes": det["boxes"][:0], "scores": det["scores"][:0], "labels": det["labels"][:0]}
            boxes = det["boxes"] * self.window_scale + offset.to(det["boxes"].device).repeat(2)
            results.append({"boxes": boxes.cpu(), "scores": det["scores"].cpu(), "labels": det["labels"].cpu()})
        return results


def load_module(model_cfg: dict, train_cfg: dict | None = None, score_threshold: float = 0.4) -> RTDetrLightning:
    """Build the module from the model config: a hub id, a local HF folder or a Lightning .ckpt."""
    source = model_cfg.get("checkpoint") or model_cfg["hub_id"]
    module = RTDetrLightning(model_cfg["hub_id"] if str(source).endswith(".ckpt") else source,
                             model_cfg["class_names"], train_cfg=train_cfg,
                             image_size=model_cfg["image_size"], score_threshold=score_threshold)
    if str(source).endswith(".ckpt"):
        state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state)
        print(f"loaded Lightning checkpoint {source}")
    return module


def download_model(hub_id: str, out_dir: str | Path) -> Path:
    """Save a Hugging Face RT-DETR checkpoint (weights, config, processor) into a local folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    RTDetrForObjectDetection.from_pretrained(hub_id).save_pretrained(out)
    AutoImageProcessor.from_pretrained(hub_id).save_pretrained(out)
    print(f"saved {hub_id} to {out}/")
    return out
