"""Command line entry point (cloud, GPU): DINOv2 embeddings of the tiles, saved as one .npz file.

    uv run python embed.py --tiles data/tiles_10cm --model facebook/dinov2-base \
        --out data/embeddings/10cm/dinov2_base.npz

The file holds one row per tile with the tile_id inside it (see src/embeddings.py). Download it
and put it in the same place in your local repo; fiftyone_workflow.ipynb links it by tile_id.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.annotations import index_with_paths
from src.embeddings import save_embeddings
from src.zeroshot import pick_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 embeddings of the tiles")
    parser.add_argument("--tiles", default="data/tiles_10cm", help="tiles folder of tile_cogs.py")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--out", default=None, help="output .npz (default: data/embeddings/<res>cm/<model>.npz)")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--annotated-only", action="store_true", help="only tiles with a GeoJSON")
    # same preprocessing as notebook 08: the whole tile, resized to 644 px (46 x 46 patches of 14 px), no crop.
    # The default DINOv2 processor would resize to 256 and centre-crop to 224, dropping the tile edges.
    parser.add_argument("--input-px", type=int, default=644, help="side of the resized tile (multiple of 14)")
    args = parser.parse_args()
    if args.input_px % 14:
        raise SystemExit("--input-px must be a multiple of 14 (the DINOv2 patch size)")

    index = index_with_paths(args.tiles)
    if args.annotated_only:
        labels = Path(args.tiles) / "labels"
        index = index[[(labels / f"{t}.geojson").exists() for t in index["tile_id"]]]
    if index.empty:
        raise SystemExit("no tiles to embed")
    res = float((index["east"] - index["west"]).iloc[0] / args.tile)
    crs = str(index["crs"].iloc[0])
    model_name = args.model.split("/")[-1].replace("-", "_")
    out = Path(args.out or f"data/embeddings/{round(res * 100)}cm/{model_name}.npz")

    device = pick_device()
    print(f"device: {device}  model: {args.model}  tiles: {len(index)}  resolution: {res:.3f} m -> {out}")
    processor = AutoImageProcessor.from_pretrained(
        args.model, size={"height": args.input_px, "width": args.input_px}, do_center_crop=False)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    rows = list(index.itertuples(index=False))
    vectors = []
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        images = [Image.open(r.tile_file).convert("RGB") for r in chunk]
        inputs = processor(images=images, return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0]          # the [CLS] token of each tile
        vectors.append(cls.float().cpu().numpy())
        print(f"  {min(i + args.batch, len(rows))}/{len(rows)}", end="\r")
    print()

    save_embeddings(
        out, np.concatenate(vectors),
        tile_ids=[r.tile_id for r in rows], scene_ids=[r.scene_id for r in rows],
        bounds=np.array([[r.west, r.south, r.east, r.north] for r in rows]),
        crs=crs, resolution_m=res, tile_size=args.tile, model=args.model, input_px=args.input_px,
    )
    print(f"saved {len(rows)} embeddings of dimension {vectors[0].shape[1]} -> {out}")


if __name__ == "__main__":
    main()
