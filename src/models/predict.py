# -*- coding: utf-8 -*-
import os
import cv2
import wandb
import click
import logging
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from metrics import r2_score
from src.data import config
from tensorflow.keras.utils import CustomObjectScope
from keras.metrics import mean_squared_error, mean_absolute_error


@click.command()
@click.argument("test_path", default=config.TEST_PATH, type=click.Path(exists=True))
@click.argument("models_path", default=config.MODELS_PATH, type=click.Path())
@click.argument("inference_path", default=config.INFERENCE_PATH, type=click.Path())
@click.option("--data", default=config.DATA_NAME, help="type of data")
def predict(test_path: str, models_path: str, inference_path: str, data: str) -> None:
    """
    Predict
    :param test_path:
    :param models_path:
    :param inference_path:
    :param data:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Prediction")

    # download model from artifact
    model_name = f"{data}"

    # Predict the model on the test data
    """ Loading model """
    with CustomObjectScope(
        {
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
        }
    ):
        images = sorted(glob(os.path.join(test_path, "*.png")))
        js = sorted(glob(os.path.join(test_path, "*.json")))

        for i, j in tqdm(zip(images, js), total=len(images)):
            # read image
            name = i.split("/")[-1]
            x = cv2.imread(i)
            resized = cv2.resize(
                x,
                (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
                interpolation=cv2.INTER_AREA,
            )
            x = resized / 255.0
            x = x.astype(np.float32)

            x = x[np.newaxis, :, :, :]

            """Predict"""
            model = tf.keras.models.load_model(
                os.path.join(models_path, f"{model_name}.h5")
            )
            y_pred = model.predict(x)
            print(y_pred)
            width, height = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
            start_point_1, end_point_1 = (round(width * y_pred[0, 0]), 0), (
                round(width * y_pred[0, 0]),
                height,
            )
            start_point_2, end_point_2 = (round(width * y_pred[0, 1]), 0), (
                round(width * y_pred[0, 1]),
                height,
            )

            # Green color in BGR
            color = (0, 255, 0)

            # Line thickness of 9 px
            thickness = 1

            # # Using cv2.line() method
            # # Draw a diagonal green line with thickness of 9 px
            resized = cv2.line(resized, start_point_1, end_point_1, color, thickness)
            resized = cv2.line(resized, start_point_2, end_point_2, color, thickness)

            cv2.imwrite(os.path.join(inference_path, name), resized)



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    predict()
