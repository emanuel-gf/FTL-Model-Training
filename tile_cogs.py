"""Command line entry point: resample and tile the COGs of a folder.

    uv run python tile_cogs.py --input cog --gsd 0.10 --tile 640 --overlap 128 --output data/tiles
"""

import argparse
from pathlib import Path

from src.tiler import tile_cogs


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample (optional) and tile COG scenes")
    parser.add_argument("--input", type=Path, default=Path("cog"), help="folder with the scene COGs")
    parser.add_argument("--gsd", type=float, default=None, help="resample to this ground resolution (m), e.g. 0.10")
    parser.add_argument("--tile", type=int, default=640, help="tile size in pixels")
    parser.add_argument("--overlap", type=int, default=0, help="overlap between tiles in pixels")
    parser.add_argument("--min-valid", type=float, default=0.9, help="skip tiles with less data than this")
    parser.add_argument("--output", type=Path, default=Path("data/tiles"), help="output folder")
    parser.add_argument("--virtual", action="store_true", help="also build one VRT per scene (needs GDAL)")
    parser.add_argument("--overwrite", action="store_true", help="redo resampled COGs that exist")
    args = parser.parse_args()

    tile_cogs(args.input, args.output, gsd=args.gsd, tile=args.tile, overlap=args.overlap,
              min_valid=args.min_valid, virtual=args.virtual, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
