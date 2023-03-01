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
from src.data.make_dataset import draw_line, image_resized
from keras.utils import CustomObjectScope
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
            x, resized = image_resized(i, config.IMAGE_SIZE)

            """Predict"""
            model = tf.keras.models.load_model(
                os.path.join(models_path, f"{data}.h5")
            )
            y_pred = model.predict(x)
            print(y_pred)

            pred_image = draw_line(
                config.IMAGE_SIZE,
                name,
                y_pred[0, 0],
                y_pred[0, 1],
                resized,
                inference_path,
            )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    predict()
