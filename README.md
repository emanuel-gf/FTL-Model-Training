# RT-DETR building detection: download, fine-tune, infer

Self-contained folder for training and running RT-DETR on OpenAerialMap imagery. It runs on
**Google Colab**, **Kaggle** or a local machine, and every command goes through **uv**.

- **Training and testing** are PyTorch Lightning modules (`src/model.py`, `src/dataset.py`), run by a Lightning `Trainer`.
- **Settings** are Hydra configs in `configs/` and can be overridden on the command line.
- **Inference** reads 640 × 640 rasterio windows from each COG, predicts through Lightning, and writes one GeoJSON per class (lon/lat).

## Layout

```
colab_training/
├── rtdetr_building_pipeline.ipynb   # the notebook: runs every step
├── download_images.py               # uv run python download_images.py --list list_of_ids.txt --out cog
├── download_model.py                # uv run python download_model.py --hub PekingU/rtdetr_r50vd --out models/rtdetr_r50
├── tile_cogs.py                     # uv run python tile_cogs.py --input cog --gsd 0.10 --tile 640 --output data/tiles
├── train.py                         # uv run python train.py train.epochs=10   (Hydra + Lightning)
├── infer.py                         # uv run python infer.py inference.cog=cog/<id>.tif   (Hydra + Lightning)
├── list_of_ids.txt                  # the OpenAerialMap items to download (committed)
├── src/annotations.py               # labelme (10 cm) -> one GeoJSON per 5 cm tile (data/tiles_5cm/labels/)
├── src/fo_dataset.py                # 5 cm / 640 px tiles -> FiftyOne dataset with building ground truth
├── src/zeroshot.py                  # RT-DETR, OWLv2 and SAM2 zero-shot runs + FiftyOne evaluation
├── configs/
│   ├── config.yaml                  # root config
│   ├── model/rtdetr_r50.yaml        # hub id or local checkpoint, classes, image size
│   ├── data/buildings.yaml          # labelled tiles, label -> class map, split by scene
│   ├── train/finetune.yaml          # batch size, learning rates, epochs, freezing, precision
│   └── inference/cog.yaml           # COG, resolution, window, stride, threshold, NMS
├── src/
│   ├── download.py                  # COG download from the STAC API + manifest.csv
│   ├── tiler.py                     # resampling and tiling (self-contained)
│   ├── labels.py                    # labelme JSON -> boxes (clipped, min area ratio)
│   ├── dataset.py                   # tile dataset, augmentation, Lightning DataModule
│   ├── model.py                     # RT-DETR Lightning module, freezing, decoding, mAP
│   └── inference.py                 # windowed COG inference, NMS, GeoJSON
├── cog/                             # downloaded COGs (not committed)
├── data/labelme/images/             # labelled tiles: .tif + labelme .json (committed, small)
└── outputs/                         # Hydra runs: checkpoints, logs, split.csv, hf_model/, geojson/
```

## Run it

**Colab or Kaggle:** open `rtdetr_building_pipeline.ipynb`, set `REPO_URL` in step 1 to this
repository, enable a GPU (Colab: *Runtime → Change runtime type*; Kaggle: *Settings → Accelerator*), and run the cells.

**Local (macOS/Linux):**

```bash
cd colab_training
uv sync
uv run python download_model.py --hub PekingU/rtdetr_r50vd --out models/rtdetr_r50
uv run python download_images.py --list list_of_ids.txt --out cog
uv run python train.py train.epochs=10
uv run python infer.py model.checkpoint=outputs/<run>/hf_model inference.cog=cog/<id>.tif
```

## Settings to know

| Setting | Where | Default | Why |
|---|---|---|---|
| Ground resolution | `inference.gsd`, `tile_cogs.py --gsd` | 0.10 m/px | Training and inference must use the same resolution |
| Tile / window size | `model.image_size`, `inference.window` | 640 | Matches the tiles used for training |
| Stride | `inference.stride` | 512 | 128 px overlap between windows |
| Class mapping | `data.label_map` | `building`, `building-ext` → building; `lote` ignored | Parcels are not buildings |
| Scene split | `data.val_fraction`, `data.test_fraction` | 0.2 / 0.1 | Split by scene, so no scene is in two splits |
| Freezing | `train.freeze_backbone`, `train.freeze_encoder`, `train.freeze_decoder_layers` | off / off / 0 | Which initial parts stay fixed |
| Batch size | `train.batch_size`, `train.accumulate_grad_batches` | 4 / 1 | Lower the batch size if the GPU runs out of memory |

## Outputs

- `outputs/<timestamp>/checkpoints/best.ckpt`: the best Lightning checkpoint (val/map_50, or train/loss without a validation split)
- `outputs/<timestamp>/hf_model/`: the same weights as a Hugging Face folder, for `infer.py`
- `outputs/<timestamp>/logs/`: metrics per step and per epoch (CSV)
- `outputs/<timestamp>/split.csv`: which tile is in which split
- `outputs/geojson/<scene>_<class>.geojson`: detections in lon/lat (WGS84), with score, scene and area

## Licences

The COGs are CC BY 4.0 (see `cog/manifest.csv`). Credit the producer when you publish results, and check the licence before sharing a trained model.
