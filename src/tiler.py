"""Resampling and tiling of GeoTIFF scenes.

This is the self-contained version of the project's tiler.py:
  - resample_to_cog: change the ground resolution and save a Cloud-Optimized GeoTIFF
  - axis_starts / tile_scene: square tiles with stride (shift-inward edge rule, no padding)
  - tile_cogs: resample (optional) and tile every scene of a folder
  - build_vrt: one virtual raster (VRT) per scene, needs the GDAL command-line tools
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds


def ground_resolution_m(src: rasterio.DatasetReader) -> tuple[float, float]:
    """Pixel size in metres (x, y). Geographic CRSs are converted at the image centre latitude."""
    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
    if src.crs.is_geographic:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        lat = math.radians((north + south) / 2)
        res_x *= 111_320 * math.cos(lat)
        res_y *= 110_574
    return res_x, res_y


def write_cog(data, transform, crs, nodata, dst: Path) -> None:
    """Write a numpy array (bands, rows, cols) as a tiled, deflate-compressed COG."""
    count, height, width = data.shape
    profile = dict(driver="GTiff", dtype=data.dtype, count=count, height=height, width=width,
                   crs=crs, transform=transform, nodata=nodata)
    with MemoryFile() as mem:
        with mem.open(**profile, tiled=True, blockxsize=512, blockysize=512) as tmp:
            tmp.write(data)
        with mem.open() as tmp:
            rasterio.shutil.copy(tmp, dst, driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=512)


def resample_to_cog(src_path: Path, dst: Path, gsd: float) -> Path:
    """Resample a scene to `gsd` metres per pixel (area averaging) and save it as a COG."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        gx, gy = ground_resolution_m(src)
        out_h = int(round(src.height * gy / gsd))
        out_w = int(round(src.width * gx / gsd))
        data = src.read(out_shape=(src.count, out_h, out_w), resampling=Resampling.average)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        write_cog(data, transform, src.crs, 0, dst)
    return dst


def axis_starts(size: int, tile: int, stride: int) -> list[int]:
    """Start positions along one axis; the last tile is shifted so the far edge is always covered."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_scene(src_path: Path, tiles_dir: Path, tile: int, stride: int, min_valid: float) -> list[dict]:
    """Cut one scene into tiles. Tiles with less than `min_valid` data are skipped."""
    rows: list[dict] = []
    scene = src_path.stem
    with rasterio.open(src_path) as src:
        gsd = src.res[0]
        for r, y0 in enumerate(axis_starts(src.height, tile, stride)):
            for c, x0 in enumerate(axis_starts(src.width, tile, stride)):
                win = Window(x0, y0, tile, tile)
                data = src.read([1, 2, 3], window=win)
                valid = float((data.max(axis=0) > 0).mean())
                if valid < min_valid:
                    continue

                tile_id = f"{scene}_r{r:02d}_c{c:02d}"
                tile_file = tiles_dir / f"{tile_id}.tif"
                with rasterio.open(tile_file, "w", driver="GTiff", height=tile, width=tile, count=3,
                                   dtype=data.dtype, crs=src.crs, transform=src.window_transform(win),
                                   compress="deflate", tiled=True, blockxsize=256, blockysize=256) as out:
                    out.write(data)

                west, south, east, north = window_bounds(win, src.transform)
                lon, lat = warp_transform(src.crs, "EPSG:4326", [(west + east) / 2], [(south + north) / 2])
                rows.append({
                    "tile_id": tile_id, "scene_id": scene,
                    "grid_row": r, "grid_col": c, "row_px": y0, "col_px": x0,
                    "width_px": tile, "height_px": tile, "gsd_m": round(gsd, 4),
                    "crs": src.crs.to_string(), "valid_fraction": round(valid, 4),
                    "west": round(west, 6), "south": round(south, 6),
                    "east": round(east, 6), "north": round(north, 6),
                    "centre_lon": round(lon[0], 6), "centre_lat": round(lat[0], 6),
                    "tile_file": str(tile_file), "source_file": str(src_path),
                })
    return rows


def build_vrt(src_paths: list[Path], vrt_path: Path) -> Path:
    """Build a VRT with gdalbuildvrt (GDAL command-line tools must be installed)."""
    gdalbuildvrt = shutil.which("gdalbuildvrt")
    if gdalbuildvrt is None:
        raise RuntimeError("gdalbuildvrt not found. Install GDAL (apt install gdal-bin, brew install gdal, or QGIS).")
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([gdalbuildvrt, "-overwrite", "-srcnodata", "0", "-vrtnodata", "0",
                    str(vrt_path), *map(str, src_paths)], check=True)
    return vrt_path


def tile_cogs(input_dir: str | Path, output_dir: str | Path, gsd: float | None = None,
              tile: int = 640, overlap: int = 0, stride: int | None = None,
              min_valid: float = 0.9, virtual: bool = False,
              resampled_dir: str | Path | None = None, overwrite: bool = False) -> Path:
    """Resample (optional) and tile every .tif of `input_dir`. Returns the tile index CSV.

    Output layout:  <output_dir>/tiles/*.tif, <output_dir>/tiles_index.csv, and with virtual=True
    <output_dir>/vrt/<scene>.vrt. Resampled scenes go to `resampled_dir` (default <input_dir>/resampled).
    """
    stride = stride if stride is not None else tile - overlap
    if not 0 < stride <= tile:
        raise ValueError(f"stride must be between 1 and the tile size ({tile}); got {stride}")

    scenes = sorted(Path(input_dir).glob("*.tif"))
    if not scenes:
        raise FileNotFoundError(f"No .tif scenes in {input_dir}")

    sources = scenes
    if gsd is not None:
        res_dir = Path(resampled_dir) if resampled_dir else Path(input_dir) / "resampled"
        sources = []
        for scene in scenes:
            dst = res_dir / scene.name
            if not dst.exists() or overwrite:
                print(f"resampling to {gsd * 100:g} cm: {scene.name}")
                resample_to_cog(scene, dst, gsd)
            sources.append(dst)

    out = Path(output_dir)
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for src in sources:
        scene_rows = tile_scene(src, tiles_dir, tile, stride, min_valid)
        rows.extend(scene_rows)
        print(f"{src.name}: {len(scene_rows)} tiles")

    if virtual:
        for src in sources:
            build_vrt([src], out / "vrt" / f"{src.stem}.vrt")

    index_path = out / "tiles_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"{len(rows)} tiles from {len(sources)} scenes -> {out}/ (index: {index_path.name})")
    return index_path
