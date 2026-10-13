"""Turn a raw 4-band Sentinel-2 GeoTIFF into model input features."""
import numpy as np
import rasterio

REFLECTANCE_CAP = 0.3  # typical land reflectance stays below this; used for scaling


def build_features(path: str):
    """Return (features[H, W, 5], valid[H, W]).

    Channels: B2, B3, B4, B8 (scaled to 0-1) + NDVI (rescaled to 0-1).
    `valid` is False where the composite has no data (all dates cloud-masked).
    """
    with rasterio.open(path) as src:
        arr = src.read().astype("float32") / 10000.0  # (4, H, W)

    valid = (arr > 0).all(axis=0)
    red, nir = arr[2], arr[3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = (np.clip(ndvi, -1, 1) + 1) / 2

    bands = np.clip(arr / REFLECTANCE_CAP, 0, 1)
    feats = np.concatenate([bands, ndvi[None]], axis=0)
    feats[:, ~valid] = 0
    return np.transpose(feats, (1, 2, 0)).astype("float32"), valid
