"""Build an interactive HTML map of forest loss.

Usage:
    python -m deforestation.visualize --config config.yaml
Open outputs/forest_loss_map.html in a browser.
"""
import argparse
import json
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from deforestation.config import load_config


def mask_overlay(path, rgb, downsample=4):
    """Reproject a forest mask to WGS84 and return (RGBA uint8 array, folium bounds)."""
    with rasterio.open(path) as src:
        transform, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        dst = np.zeros((h, w), dtype="uint8")
        reproject(
            rasterio.band(src, 1), dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
    west, south, east, north = array_bounds(h, w, transform)
    dst = dst[::downsample, ::downsample]
    rgba = np.zeros((*dst.shape, 4), dtype="uint8")
    rgba[dst == 1] = (*rgb, 140)
    return rgba, [[south, west], [north, east]]


def build_map(cfg, out_html):
    out = Path(cfg["paths"]["outputs"])
    loss = gpd.read_file(out / "forest_loss.geojson")
    stats = json.loads((out / "stats.json").read_text())

    west, south, east, north = cfg["aoi"]
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)

    for key, color in (("A", (34, 139, 34)), ("B", (0, 100, 0))):
        rgba, bounds = mask_overlay(out / f"forest_mask_{key}.tif", color)
        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=bounds, name=f"Forest cover {cfg['labels'][key]}", show=(key == "A"),
        ).add_to(m)

    if len(loss):
        folium.GeoJson(
            loss,
            name="Newly cleared areas",
            style_function=lambda _: {"color": "#d62728", "weight": 1, "fillColor": "#d62728", "fillOpacity": 0.6},
            tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=["Cleared (ha):"], localize=True),
        ).add_to(m)
    else:
        print("No cleared areas detected; map will show forest layers only.")
    folium.LayerControl(collapsed=False).add_to(m)

    panel = f"""
    <div style="position: fixed; top: 12px; right: 12px; z-index: 9999; background: white; padding: 10px 14px;
                border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.3); font: 13px/1.5 sans-serif;">
      <b>Forest change</b><br>{cfg['labels']['A']} &rarr; {cfg['labels']['B']}<br>
      Cleared: <b>{stats['forest_loss_ha']:,} ha</b> ({stats['forest_loss_percent_of_A']}% of forest)<br>
      Cleared patches: {stats['n_cleared_patches']}
    </div>"""
    m.get_root().html.add_child(folium.Element(panel))
    m.save(out_html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_html = Path(cfg["paths"]["outputs"]) / "forest_loss_map.html"
    build_map(cfg, out_html)
    print(f"Saved {out_html}")


if __name__ == "__main__":
    main()
