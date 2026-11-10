"""Download the Cloud-Optimized GeoTIFFs (COGs) of a list of OpenAerialMap items.

The items are resolved through the HOT OpenAerialMap STAC API. Each file is saved as
<out_dir>/<item_id>.tif, and <out_dir>/manifest.csv records the licence, producer and
capture date of every item.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pystac_client
import requests
from tqdm.auto import tqdm

STAC_API = "https://api.imagery.hotosm.org/stac/"
COLLECTION = "openaerialmap"
MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = ["item_id", "status", "path", "size_mb", "gsd_m", "captured", "producer", "license"]


def read_ids(list_path: str | Path) -> list[str]:
    """Item IDs from a text file: one ID or item URL per line, '#' starts a comment."""
    ids: list[str] = []
    for raw in Path(list_path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().rstrip("/")
        if not line:
            continue
        item_id = line.split("/")[-1]
        if item_id not in ids:
            ids.append(item_id)
    if not ids:
        raise ValueError(f"No item IDs found in {list_path}")
    return ids


def _stream(url: str, dest: Path, expected_size: int | None, overwrite: bool, retries: int = 3) -> str:
    """Stream one file to disk. Returns 'downloaded', 'skipped' or 'failed: <reason>'."""
    if dest.exists() and not overwrite:
        if expected_size is None or dest.stat().st_size == expected_size:
            return "skipped"
    tmp = dest.with_suffix(".part")
    last_error = ""
    for _ in range(retries):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
            tmp.replace(dest)
            return "downloaded"
        except requests.RequestException as err:
            last_error = str(err)
    if tmp.exists():
        tmp.unlink()
    return f"failed: {last_error}"


def download_cogs(ids: list[str], out_dir: str | Path = "cog", overwrite: bool = False) -> Path:
    """Download the 'visual' COG of every item and write the manifest. Returns the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = pystac_client.Client.open(STAC_API)
    collection = client.get_collection(COLLECTION)

    rows = []
    for item_id in tqdm(ids, desc="Downloading COGs"):
        item = collection.get_item(item_id)
        if item is None:
            rows.append({"item_id": item_id, "status": "failed: item not found in STAC"})
            continue

        asset = item.assets["visual"]
        dest = out / f"{item_id}.tif"
        expected = asset.extra_fields.get("file:size")
        status = _stream(asset.href, dest, expected, overwrite)

        props = item.properties
        producer = props.get("oam:producer_name") or ""
        rows.append({
            "item_id": item_id,
            "status": status,
            "path": str(dest),
            "size_mb": round(dest.stat().st_size / 1e6, 1) if dest.exists() else None,
            "gsd_m": props.get("gsd"),
            "captured": props.get("start_datetime"),
            "producer": producer,
            "license": props.get("license"),
        })

    manifest = out / MANIFEST_NAME
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        key = row["status"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"COGs in {out}/: {counts} | manifest: {manifest}")
    return manifest
