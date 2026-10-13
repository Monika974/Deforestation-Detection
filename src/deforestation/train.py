"""Train the U-Net and evaluate it on held-out spatial blocks.

Usage:
    python -m deforestation.train --config config.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from deforestation.config import load_config
from deforestation.losses import bce_dice_loss, dice_coef
from deforestation.model import build_unet


def augment(x, y):
    """Same random flip / rotation applied to image and mask."""
    both = tf.concat([x, y], axis=-1)
    both = tf.image.random_flip_left_right(both)
    both = tf.image.random_flip_up_down(both)
    both = tf.image.rot90(both, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return both[..., :-1], both[..., -1:]


def make_dataset(X, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), tf.cast(b, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(X)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def pixel_metrics(y_true, prob, thr=0.5):
    pred = prob >= thr
    true = y_true.astype(bool)
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "iou": float(tp / (tp + fp + fn + 1e-9)),
        "f1": float(2 * precision * recall / (precision + recall + 1e-9)),
        "precision": float(precision),
        "recall": float(recall),
    }


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[1].plot(history.history["iou_forest"], label="train")
    axes[1].plot(history.history["val_iou_forest"], label="val")
    axes[1].set_title("Forest IoU")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(X, y, prob, path, n=4):
    n = min(n, len(X))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        rgb = np.clip(X[i][..., [2, 1, 0]].astype("float32") * 1.5, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(y[i, ..., 0], cmap="Greens", vmin=0, vmax=1)
        axes[i, 2].imshow(prob[i, ..., 0] >= 0.5, cmap="Greens", vmin=0, vmax=1)
        for ax, title in zip(axes[i], ["Sentinel-2 RGB", "Ground truth", "Prediction"]):
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]
    keras.utils.set_random_seed(t["seed"])

    data = np.load(Path(cfg["paths"]["processed"]) / "patches.npz")
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]

    model = build_unet(X_tr.shape[1:], base=t["base_filters"], depth=t["depth"])
    model.compile(
        optimizer=keras.optimizers.Adam(t["lr"]),
        loss=bce_dice_loss,
        metrics=[dice_coef, keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou_forest")],
    )

    models_dir = Path(cfg["paths"]["models"])
    figs_dir = Path(cfg["paths"]["figures"])
    models_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(models_dir / "unet.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    history = model.fit(
        make_dataset(X_tr, y_tr, t["batch_size"], True),
        validation_data=make_dataset(X_va, y_va, t["batch_size"], False),
        epochs=t["epochs"],
        callbacks=callbacks,
    )
    plot_history(history, figs_dir / "training_curves.png")

    if len(X_te):
        prob = model.predict(X_te.astype("float32"), batch_size=t["batch_size"])
        metrics = pixel_metrics(y_te, prob)
        metrics["n_test_patches"] = int(len(X_te))
        (Path(cfg["paths"]["outputs"]) / "metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_predictions(X_te, y_te, prob, figs_dir / "test_predictions.png")
        print("Test metrics:", json.dumps(metrics, indent=2))
    else:
        print("No test patches (AOI too small for the spatial split); enlarge the AOI or reduce split_block.")


if __name__ == "__main__":
    main()
