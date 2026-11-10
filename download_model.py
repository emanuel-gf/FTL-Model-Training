"""Command line entry point: download an RT-DETR checkpoint into a local folder.

    uv run python download_model.py --hub PekingU/rtdetr_r50vd --out models/rtdetr_r50
"""

import argparse

from src.model import download_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an RT-DETR checkpoint from the Hugging Face hub")
    parser.add_argument("--hub", default="PekingU/rtdetr_r50vd", help="hub id, e.g. PekingU/rtdetr_r101vd")
    parser.add_argument("--out", default="models/rtdetr_r50", help="local folder")
    args = parser.parse_args()
    download_model(args.hub, args.out)


if __name__ == "__main__":
    main()
