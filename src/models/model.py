from src.data import config
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Flatten, Dense, Input


def mobilenetv3_small(height: int, width: int, channels: int) -> None:
    shape = (height, width, channels)
    inputs = Input(shape)
    backbone = MobileNetV3Small(
        weights="imagenet", alpha=0.75, input_tensor=inputs, include_top=False
    )
    for layer in backbone.layers:  # Freeze the layers
        layer.trainable = False

    x = Flatten()(backbone.output)
    x = Dense(units=64, activation="relu")(x)
    x = Dense(units=2, activation="linear")(x)

    model = Model(inputs=backbone.input, outputs=x)

    return model


if __name__ == "__main__":
    model = mobilenetv3_small(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.IMAGE_SIZE[2])
    model.summary()
