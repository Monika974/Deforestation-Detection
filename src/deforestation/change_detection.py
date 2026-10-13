"""Compare two forest masks and extract newly cleared areas.

Usage:
    python -m deforestation.change_detection --config config.yaml
"""
import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape

from deforestation.config import load_config

NODATA = 255


def min_pixels_from_ha(min_area_ha: float, pixel_size_m: float) -> int:
    return max(1, math.ceil(min_area_ha * 10_000 / pixel_size_m ** 2))


def remove_small_regions(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask.astype("uint8"), labels, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_pixels])
    return keep[labels]


def compute_loss(mask_a: np.ndarray, mask_b: np.ndarray, min_pixels: int = 50) -> np.ndarray:
    """Forest in A and non-forest in B = newly cleared. No-data pixels are ignored."""
    valid = (mask_a != NODATA) & (mask_b != NODATA)
    loss = (mask_a == 1) & (mask_b == 0) & valid
    loss = ndi.binary_opening(loss, structure=np.ones((3, 3)))  # drop 1-2 px speckle
    return remove_small_regions(loss, min_pixels)


def loss_to_polygons(loss: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Vectorise the loss mask. `crs` must be projected (metres) so areas are in hectares."""
    geoms = [
        shape(g)
        for g, v in features.shapes(loss.astype("uint8"), mask=loss, transform=transform)
        if v == 1
    ]
    gdf = gpd.GeoDataFrame({"id": range(1, len(geoms) + 1)}, geometry=geoms, crs=crs)
    gdf["area_ha"] = gdf.geometry.area / 10_000
    return gdf


def compute_stats(mask_a, mask_b, loss, pixel_size_m):
    ha = pixel_size_m ** 2 / 10_000
    valid = (mask_a != NODATA) & (mask_b != NODATA)
    forest_a = int(((mask_a == 1) & valid).sum())
    forest_b = int(((mask_b == 1) & valid).sum())
    loss_px = int(loss.sum())
    return {
        "forest_area_A_ha": round(forest_a * ha, 1),
        "forest_area_B_ha": round(forest_b * ha, 1),
        "forest_loss_ha": round(loss_px * ha, 1),
        "forest_loss_percent_of_A": round(100 * loss_px / max(forest_a, 1), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path(cfg["paths"]["outputs"])
    with rasterio.open(out / "forest_mask_A.tif") as a_src, rasterio.open(out / "forest_mask_B.tif") as b_src:
        mask_a, mask_b = a_src.read(1), b_src.read(1)
        transform, crs, profile = a_src.transform, a_src.crs, a_src.profile
    pixel = abs(transform.a)

    loss = compute_loss(mask_a, mask_b, min_pixels_from_ha(cfg["change"]["min_area_ha"], pixel))

    profile.update(dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(out / "forest_loss.tif", "w", **profile) as dst:
        dst.write(loss.astype("uint8"), 1)

    gdf = loss_to_polygons(loss, transform, crs)
    gdf.to_crs(4326).to_file(out / "forest_loss.geojson", driver="GeoJSON")

    stats = compute_stats(mask_a, mask_b, loss, pixel)
    stats["n_cleared_patches"] = int(len(gdf))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))



if __name__ == "__main__":
    main()
