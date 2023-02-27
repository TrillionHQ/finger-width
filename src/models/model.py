from src.data import config
from tensorflow.keras.models import Model
from keras.engine.functional import Functional
from keras.utils.layer_utils import count_params
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Input


def mobilenetv3_small(
    im_size: int, channels: int, units1: int, alpha: int
) -> Functional:
    """
    MobileNetV3 Small
    :param im_size:
    :param channels:
    :param units1:
    :param alpha: The role of the width multiplier α is to thin a network uniformly at each layer. For a given layer
    and width multiplier α, the number of input channels M becomes αM and the number of output channels N becomes αN
    :return:
    """
    shape = (im_size, im_size, channels)
    inputs = Input(shape)

    backbone = MobileNetV3Small(
        weights="imagenet", input_tensor=inputs, alpha=alpha, include_top=False
    )
    for layer in backbone.layers:  # Freeze the layers
        layer.trainable = False

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(units=units1, activation="relu")(x)
    x = Dense(units=2, activation="linear")(x)

    model = Model(inputs=backbone.input, outputs=x)

    return model


if __name__ == "__main__":
    model = mobilenetv3_small(
        config.IMAGE_SIZE,
        3,
        config.default_config["size_layer1"],
        config.default_config["alpha"],
    )

    trainable_params = count_params(model.trainable_weights)
    non_trainable_params = count_params(model.non_trainable_weights)
    print(non_trainable_params, trainable_params)
