from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Input


def mobilenetv3_small(height: int, width: int, channels: int) -> None:
    shape = (height, width, channels)
    inputs = Input(shape)
    backbone = MobileNetV3Small(
        weights="imagenet", include_top=False, input_tensor=inputs
    )
    x_in = backbone.layers[-6].output
    print(backbone.layers[-6].output)
    x = Dense(units=1024, activation="relu")(x_in)
    x = Dense(units=512, activation="relu")(x)
    x = Dense(units=2, activation="linear")(x)

    model = Model(inputs, x)
    print(type(model))
    return model


if __name__ == "__main__":
    model = mobilenetv3_small(224, 224, 3)
    model.summary()
