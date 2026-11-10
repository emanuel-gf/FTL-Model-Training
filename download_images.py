"""Command line entry point: download the COGs listed in a text file.

    uv run python download_images.py --list list_of_ids.txt --out cog
"""

import argparse

from src.download import download_cogs, read_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenAerialMap COGs from a list of item IDs")
    parser.add_argument("--list", required=True, help="text file with one item ID or URL per line")
    parser.add_argument("--out", default="cog", help="folder for the COGs (default: cog/)")
    parser.add_argument("--overwrite", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()

    ids = read_ids(args.list)
    print(f"{len(ids)} unique items in {args.list}")
    download_cogs(ids, out_dir=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
