from src.data import config
from tensorflow.keras.models import Model
from keras.utils.layer_utils import count_params
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Input


def mobilenetv3_small(height: int, width: int, channels: int, units1: int) -> None:
    shape = (height, width, channels)
    inputs = Input(shape)
    backbone = MobileNetV3Small(
        weights="imagenet", alpha=0.75, input_tensor=inputs, include_top=False
    )
    for layer in backbone.layers:  # Freeze the layers
        layer.trainable = False

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(units=units1, activation="relu")(x)
    # x = Dense(units=units2, activation="relu")(x)
    x = Dense(units=2, activation="linear")(x)

    model = Model(inputs=backbone.input, outputs=x)

    return model


if __name__ == "__main__":
    model = mobilenetv3_small(
        config.IMAGE_SIZE[0],
        config.IMAGE_SIZE[1],
        config.IMAGE_SIZE[2],
        config.default_config["size_layer1"],
        # config.default_config["size_layer2"],
    )
    model.summary()

    trainable_params = count_params(model.trainable_weights)
    non_trainable_params = count_params(model.non_trainable_weights)
    print(non_trainable_params, trainable_params)
