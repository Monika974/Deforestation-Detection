"""Predict forest masks for the two comparison dates (A and B).

Usage:
    python -m deforestation.predict --config config.yaml

Output masks: 1 = forest, 0 = non-forest, 255 = no data (cloud-masked in all scenes).
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from tensorflow import keras

from deforestation.config import load_config
from deforestation.features import build_features


def predict_proba(model, feats, size=256, stride=128, batch=8):
    """Sliding-window inference; overlapping windows are averaged to hide tile seams."""
    H, W, _ = feats.shape
    ph = size - H if H < size else (-(H - size)) % stride
    pw = size - W if W < size else (-(W - size)) % stride
    padded = np.pad(feats, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H2, W2, _ = padded.shape

    acc = np.zeros((H2, W2), dtype="float32")
    cnt = np.zeros((H2, W2), dtype="float32")
    coords = [(r, c) for r in range(0, H2 - size + 1, stride) for c in range(0, W2 - size + 1, stride)]

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = np.stack([padded[r:r + size, c:c + size] for r, c in chunk])
        p = model.predict(x, verbose=0)[..., 0]
        for (r, c), pp in zip(chunk, p):
            acc[r:r + size, c:c + size] += pp
            cnt[r:r + size, c:c + size] += 1
    return (acc / cnt)[:H, :W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_path = args.model or Path(cfg["paths"]["models"]) / "unet.keras"
    model = keras.models.load_model(model_path, compile=False)
    raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)

    for key in ("A", "B"):
        src_path = raw / f"{key}_image.tif"
        feats, valid = build_features(src_path)
        prob = predict_proba(model, feats, cfg["patch_size"], cfg["predict"]["stride"])
        mask = np.where(valid, (prob >= cfg["predict"]["threshold"]).astype("uint8"), 255).astype("uint8")

        with rasterio.open(src_path) as src:
            profile = src.profile
        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(out / f"forest_mask_{key}.tif", "w", **profile) as dst:
            dst.write(mask, 1)
        print(f"Saved forest_mask_{key}.tif | forest share of valid pixels: {(mask[valid] == 1).mean():.1%}")


if __name__ == "__main__":
    main()
