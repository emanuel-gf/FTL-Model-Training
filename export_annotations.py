"""Command line entry point: export the labelme annotations as one GeoJSON per 10 cm tile.

    uv run python export_annotations.py --index data/tiles_10cm/tiles_index.csv \
        --labels data/labelme/images --out data/tiles_10cm/labels
"""

import argparse

from src.annotations import export_tile_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="labelme annotations -> one GeoJSON per tile")
    parser.add_argument("--index", default="data/tiles_10cm/tiles_index.csv", help="tile index of tile_cogs.py")
    parser.add_argument("--labels", default="data/labelme/images", help="folder with labelme .tif + .json pairs")
    parser.add_argument("--out", default="data/tiles_10cm/labels", help="output folder (rebuilt each run)")
    parser.add_argument("--tile", type=int, default=640, help="tile size in pixels")
    args = parser.parse_args()
    export_tile_annotations(args.index, labels_dir=args.labels, out_dir=args.out, tile=args.tile)


if __name__ == "__main__":
    main()
