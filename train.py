"""Hydra entry point: fine-tune RT-DETR on labelled tiles with PyTorch Lightning.

Training, validation and test are run by a Lightning Trainer. The best checkpoint
(by val/map_50, or train/loss when there is no validation split) is used for the test
and exported to a Hugging Face folder that infer.py can load.

    uv run python train.py
    uv run python train.py train.epochs=10 train.batch_size=2 train.freeze_backbone=true
    uv run python train.py data.labels_dir=data/labelme/images train.limit_batches=3   # smoke test
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.dataset import TileDataModule
from src.model import load_module


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    tcfg = OmegaConf.to_container(cfg.train, resolve=True)
    tcfg["seed"] = cfg.seed
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    dm = TileDataModule(OmegaConf.to_container(cfg.data, resolve=True), tcfg,
                        model_cfg["class_names"], model_cfg["image_size"])
    dm.setup("fit")
    print(f"tiles: train {len(dm.train_ds)} | val {len(dm.val_ds)} | test {len(dm.test_ds)}")
    print(f"split table: {dm.save_split(out_dir / 'split.csv')}")

    module = load_module(model_cfg, train_cfg=tcfg)
    has_val = len(dm.val_ds) > 0
    monitor, mode = ("val/map_50", "max") if has_val else ("train/loss", "min")

    callbacks = [
        ModelCheckpoint(dirpath=out_dir / "checkpoints", monitor=monitor, mode=mode,
                        filename="best", save_last=True, save_top_k=1),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if has_val and tcfg["patience"] > 0:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=tcfg["patience"]))

    trainer = L.Trainer(
        max_epochs=tcfg["epochs"],
        accelerator=tcfg["accelerator"],
        devices=tcfg["devices"],
        precision=tcfg["precision"],
        accumulate_grad_batches=tcfg["accumulate_grad_batches"],
        gradient_clip_val=tcfg["gradient_clip_val"],
        log_every_n_steps=tcfg["log_every_n_steps"],
        limit_train_batches=tcfg["limit_batches"] or 1.0,
        limit_val_batches=tcfg["limit_batches"] or 1.0,
        limit_test_batches=tcfg["limit_batches"] or 1.0,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        default_root_dir=str(out_dir),
        num_sanity_val_steps=0 if not has_val else 2,
    )
    trainer.fit(module, train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader() if has_val else None)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state)
        print(f"best checkpoint: {best_path} ({monitor}={trainer.checkpoint_callback.best_model_score})")

    if len(dm.test_ds) > 0:
        trainer.test(module, dataloaders=dm.test_dataloader())

    hf_dir = out_dir / "hf_model"
    module.model.save_pretrained(hf_dir)
    from transformers import AutoImageProcessor
    AutoImageProcessor.from_pretrained(model_cfg["hub_id"]).save_pretrained(hf_dir)
    print(f"exported Hugging Face model to {hf_dir} (use model.checkpoint={hf_dir} for inference)")


if __name__ == "__main__":
    main()
