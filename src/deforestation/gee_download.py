"""Export Sentinel-2 composites and ESA WorldCover forest labels from Google Earth Engine.

Usage:
    earthengine authenticate
    python -m deforestation.gee_download --config config.yaml

Exports land in your Google Drive folder (config: drive_folder). Download them into data/raw/.
"""
import argparse

import ee

from deforestation.config import load_config

BANDS = ["B2", "B3", "B4", "B8"]  # Blue, Green, Red, NIR


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """Keep pixels that the Scene Classification Layer does not flag as bad.

    SCL: 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow.
    """
    scl = img.select("SCL")
    good = scl.gte(2).And(scl.neq(3)).And(scl.lt(8))
    return img.updateMask(good)


def composite(aoi: ee.Geometry, start: str, end: str, max_cloud: float) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_s2_clouds)
    )
    # Median composite, stored as uint16 reflectance x 10000. 0 = no data.
    return col.select(BANDS).median().multiply(10000).toUint16().unmask(0).clip(aoi)


def forest_label(aoi: ee.Geometry) -> ee.Image:
    """Binary forest mask from ESA WorldCover 2021 (class 10 = tree cover)."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.eq(10).rename("forest").toUint8().clip(aoi)


def start_export(image, name, aoi, cfg):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=cfg["drive_folder"],
        fileNamePrefix=name,
        region=aoi,
        scale=cfg["scale"],
        crs=cfg["crs"],
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Started export: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    ee.Initialize(project=cfg["gee_project"])
    aoi = ee.Geometry.Rectangle(cfg["aoi"])

    for key, period in cfg["composites"].items():
        img = composite(aoi, period["start"], period["end"], cfg["max_cloud_percent"])
        start_export(img, f"{key}_image", aoi, cfg)

    start_export(forest_label(aoi), "train_label", aoi, cfg)
    print("\nTrack progress at https://code.earthengine.google.com/tasks")
    print("When done, download the .tif files from Drive into data/raw/")


if __name__ == "__main__":
    main()
