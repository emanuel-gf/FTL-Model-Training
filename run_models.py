"""Command line entry point: run a pretrained building detector on the tiles (no fine-tuning).

    # per-tile predictions for the annotated tiles -> outputs/predictions/<name>/<tile_id>.json
    uv run python run_models.py --kind sam2 --weights facebook/sam2.1-hiera-large --name sam2_large \
        --mode tiles --options '{"score_threshold": 0.85, "min_area_m2": 10, "max_area_m2": 1500}'

    # one GeoJSON per scene for every tile -> outputs/batch/<name>/<scene_id>.geojson
    uv run python run_models.py --kind owlv2 --weights google/owlv2-base-patch16-ensemble --name owlv2 \
        --mode scenes --options '{"score_threshold": 0.2, "queries": ["a building"]}'

--kind: rtdetr | owlv2 | sam2. --weights: hub id or local folder. --name: folder name for the outputs.
"""

import argparse
import json
from pathlib import Path

from src.zeroshot import load_detector, pick_device, predict_scenes, predict_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pretrained building detector on the tiles")
    parser.add_argument("--kind", required=True, choices=["rtdetr", "owlv2", "sam2"])
    parser.add_argument("--weights", required=True, help="hub id or local folder")
    parser.add_argument("--name", required=True, help="output folder name")
    parser.add_argument("--mode", required=True, choices=["tiles", "scenes"])
    parser.add_argument("--options", default="{}", help="detector options as JSON")
    parser.add_argument("--index", default="data/tiles_10cm/tiles_index.csv")
    parser.add_argument("--labels", default="data/tiles_10cm/labels", help="annotated tiles (mode tiles)")
    parser.add_argument("--scenes", nargs="*", default=None, help="scene ids (mode scenes), default all")
    parser.add_argument("--out", default="outputs", help="root of the outputs")
    parser.add_argument("--tile", type=int, default=640)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}  kind: {args.kind}  weights: {args.weights}  mode: {args.mode}")
    detector = load_detector(args.kind, args.weights, device, **json.loads(args.options))

    if args.mode == "tiles":
        out_dir = Path(args.out) / "predictions" / args.name
        n = predict_tiles(detector, args.index, args.labels, out_dir, tile=args.tile)
        print(f"{n} annotated tiles -> {out_dir}/")
    else:
        out_dir = Path(args.out) / "batch" / args.name
        counts = predict_scenes(detector, args.index, out_dir, scenes=args.scenes, tile=args.tile)
        print(f"buildings per scene -> {out_dir}/: {counts}")


if __name__ == "__main__":
    main()
