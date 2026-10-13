"""Segmentation loss and metrics (Dice + BCE handles forest/non-forest imbalance)."""
import tensorflow as tf
from tensorflow import keras


def dice_coef(y_true, y_pred, smooth=1.0):
    """Soft Dice coefficient; at pixel level this equals the F1 score."""
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)
