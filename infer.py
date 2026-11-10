"""Hydra entry point: detect buildings in COGs and write GeoJSON.

    uv run python infer.py model.checkpoint=models/rtdetr_r50
    uv run python infer.py inference.cog=cog/68a4a953e50784296730bf0f.tif inference.score_threshold=0.5
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.inference import detections_to_geojson, merge_detections, predict_cog, prepare_cog
from src.model import load_module


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg.inference))
    L.seed_everything(cfg.seed, workers=True)
    icfg = cfg.inference

    if icfg.cog:
        cogs = [Path(icfg.cog)]
    else:
        cogs = sorted(Path(icfg.cog_dir).glob("*.tif"))
    if not cogs:
        raise FileNotFoundError(f"No COGs found ({icfg.cog or icfg.cog_dir})")

    module = load_module(OmegaConf.to_container(cfg.model, resolve=True), score_threshold=icfg.score_threshold)
    module.eval()

    geojson_dir = Path(icfg.geojson_dir)
    summary = []
    for cog in cogs:
        target = prepare_cog(cog, icfg.gsd, Path(icfg.resampled_dir))
        result = predict_cog(module, target, window=icfg.window, stride=icfg.stride,
                             batch_size=icfg.batch_size, num_workers=icfg.num_workers,
                             min_valid=icfg.min_valid_fraction, score_threshold=icfg.score_threshold,
                             accelerator=icfg.accelerator)
        boxes, scores, labels = merge_detections(result["boxes"], result["scores"], result["labels"], icfg.nms_iou)
        files = detections_to_geojson(target, boxes, scores, labels, cfg.model.class_names, geojson_dir)
        summary.append((cog.name, result["windows"], len(boxes), [str(f) for f in files]))
        print(f"{cog.name}: {result['windows']} windows, {len(boxes)} detections -> {geojson_dir}/")

    print("\nDone:")
    for name, windows, n, files in summary:
        print(f"  {name}: {windows} windows, {n} buildings")
        for f in files:
            print(f"    {f}")


if __name__ == "__main__":
    main()
