"""U-Net for binary forest segmentation."""
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(256, 256, 5), base=32, depth=4, dropout=0.2):
    inputs = layers.Input(input_shape)
    x, skips = inputs, []

    for d in range(depth):  # encoder
        x = conv_block(x, base * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

    x = conv_block(x, base * 2 ** depth)  # bottleneck
    x = layers.Dropout(dropout)(x)

    for d in reversed(range(depth)):  # decoder
        x = layers.Conv2DTranspose(base * 2 ** d, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = conv_block(x, base * 2 ** d)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="unet")
