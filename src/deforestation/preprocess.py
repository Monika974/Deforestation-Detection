"""Cut the training scene into 256x256 patches and split them by location.

Usage:
    python -m deforestation.preprocess --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from deforestation.config import load_config
from deforestation.features import build_features


def tile(features, label, valid, size=256, min_valid=0.9):
    """Non-overlapping patches. Returns X (N,s,s,C) float16, y (N,s,s,1) uint8, pos (N,2)."""
    H, W = valid.shape
    X, Y, pos = [], [], []
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            if valid[r:r + size, c:c + size].mean() < min_valid:
                continue
            X.append(features[r:r + size, c:c + size])
            Y.append(label[r:r + size, c:c + size, None])
            pos.append((r, c))
    if not X:
        raise RuntimeError("No valid patches found. Check your composites for clouds / no-data.")
    return np.stack(X).astype("float16"), np.stack(Y).astype("uint8"), np.array(pos)


def spatial_split(pos, block=1024, fracs=(0.7, 0.15, 0.15), seed=42):
    """Assign each patch to train(0)/val(1)/test(2) by large spatial block, not at random.

    Neighbouring patches look almost identical, so a random patch split leaks information
    from train into test and inflates scores. Whole blocks are held out instead.
    """
    keys = [(int(r) // block, int(c) // block) for r, c in pos]
    blocks = sorted(set(keys))
    order = np.random.default_rng(seed).permutation(len(blocks))
    n = len(blocks)
    n_train = max(1, int(round(fracs[0] * n)))
    n_val = max(1, int(round(fracs[1] * n))) if n > 2 else 0
    assignment = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            assignment[blocks[idx]] = 0
        elif rank < n_train + n_val:
            assignment[blocks[idx]] = 1
        else:
            assignment[blocks[idx]] = 2
    return np.array([assignment[k] for k in keys])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw = Path(cfg["paths"]["raw"])
    feats, valid = build_features(raw / "train_image.tif")
    with rasterio.open(raw / "train_label.tif") as src:
        label = src.read(1)
    if label.shape != valid.shape:
        raise ValueError(f"Image {valid.shape} and label {label.shape} grids differ; re-export both with the same AOI/crs/scale.")

    X, y, pos = tile(feats, label, valid, cfg["patch_size"], cfg["min_valid_fraction"])
    split = spatial_split(pos, cfg["split_block"], seed=cfg["train"]["seed"])

    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "patches.npz",
        X_train=X[split == 0], y_train=y[split == 0],
        X_val=X[split == 1], y_val=y[split == 1],
        X_test=X[split == 2], y_test=y[split == 2],
    )
    print(f"Patches: {len(X)} | train {np.sum(split == 0)} | val {np.sum(split == 1)} | test {np.sum(split == 2)}")
    print(f"Forest fraction in labels: {y.mean():.2%}")


if __name__ == "__main__":
    main()
