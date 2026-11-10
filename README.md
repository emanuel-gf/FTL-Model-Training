# Building detection teaching project: tiles, labels, zero-shot, embeddings, fine-tuning

Two notebooks, one repository, one `pyproject.toml`:

- **`fiftyone_workflow.ipynb`** (your computer, `uv sync --group fo`, no torch): tiles and labels, FiftyOne viewer, test tags, model comparison, embedding map, new labels.
- **`model_train.ipynb`** (Colab or Kaggle, `uv sync --group gpu`): zero-shot predictions, batch inference over the scenes, DINOv2 embeddings, RT-DETR fine-tuning, inference.
- **`rtdetr_building_pipeline.ipynb`**: the earlier all-in-one notebook, kept as a reference.

Both current notebooks use the same data, explicit in their setup cell: **`data/tiles_10cm/`** (tiles) and **`data/tiles_10cm/labels/`** (GeoJSON labels). The two labelled tiles in `data/labelme/images/` are the examples the labels are made from.

## Data flow

| Step | Where | Output |
|---|---|---|
| Tile the COGs | uv (`tile_cogs.py`) | `data/tiles_10cm/tiles/*.tif` + `tiles_index.csv` |
| Export labels | uv (`export_annotations.py`) | `data/tiles_10cm/labels/<tile_id>.geojson` |
| Predictions | cloud (`run_models.py`) | `outputs/predictions/<name>/<tile_id>.json`, `outputs/batch/<name>/<scene>.geojson` |
| Embeddings | cloud (`embed.py`) | `data/embeddings/10cm/dinov2_base.npz` (vectors + `tile_id` inside) |
| Fine-tune | cloud (`train.py`) | `outputs/<run>/hf_model/` |
| Link and view | local (`fiftyone_workflow.ipynb`) | FiftyOne dataset, Brain map |

Results go from the cloud to your computer as one zip (`results_for_local.zip`), unzipped at the root of the repository. The labels go the other way through git.

## Layout

```
colab_training/
├── model_train.ipynb                # CLOUD notebook: predictions, embeddings, fine-tuning (uv sync --group gpu)
├── fiftyone_workflow.ipynb          # LOCAL notebook: FiftyOne viewer, labels, embedding map (uv sync --group fo)
├── rtdetr_building_pipeline.ipynb   # reference: the earlier all-in-one notebook
├── download_images.py               # uv run python download_images.py --list list_of_ids.txt --out cog
├── download_model.py                # uv run python download_model.py --hub PekingU/rtdetr_r50vd --out models/rtdetr_r50
├── tile_cogs.py                     # uv run python tile_cogs.py --input cog --gsd 0.10 --tile 640 --output data/tiles
├── train.py                         # uv run python train.py train.epochs=10   (Hydra + Lightning)
├── infer.py                         # uv run python infer.py inference.cog=cog/<id>.tif   (Hydra + Lightning)
├── list_of_ids.txt                  # the OpenAerialMap items to download (committed)
├── export_annotations.py            # uv run python export_annotations.py  (labelme -> one GeoJSON per tile)
├── run_models.py                    # uv run python run_models.py --kind sam2 --weights ... --name ... --mode tiles|scenes
├── embed.py                         # uv run python embed.py --tiles data/tiles_10cm   (DINOv2, cloud)
├── src/annotations.py               # labelme (10 cm) -> one GeoJSON per 10 cm tile; tile samples for training
├── src/embeddings.py                # .npz embedding files: vectors + tile_id inside (numpy only)
├── src/bootstrap.py                 # shared notebook start: uv sync for the group, path report
├── src/naming.py                    # run names shared by notebooks and scripts
├── src/zeroshot.py                  # RT-DETR, OWLv2, SAM2 detectors: per-tile JSON and per-scene GeoJSON (gpu)
├── src/fo_dataset.py                # FiftyOne dataset, predictions, evaluation, embedding link (fo group)
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
