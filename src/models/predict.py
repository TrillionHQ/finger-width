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
@click.option("--data", default=config.DATA_TYPE, help="type of data")
def main(test_path: str, models_path: str, inference_path: str, data: str) -> None:
    """
    :param processed_path:
    :param model_path:
    :param results_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="Perceptron-finger",
        job_type="predict",
    ) as run:  # config is optional here
        #
        #     # download dataset from artifact
        #     artifact = run.use_artifact(f'splited_{data}_data:latest', type='dataset')
        #     artifact_dir = artifact.download(processed_path)
        # X_test = np.load(os.path.join(processed_path, 'X_test.npy'))
        # y_test = np.load(os.path.join(processed_path, 'y_test.npy'))

        # download model from artifact
        model_name = f"{data}_40_40_model"
        artifact = run.use_artifact(f"{model_name}:v8", type="model")
        artifact_dir = artifact.download(models_path)

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

            k = 1
            for i, j in tqdm(zip(images, js), total=len(images)):
                # read image
                name = i.split("/")[-1]
                img = cv2.imread(i)
                resized = cv2.resize(
                    img,
                    (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
                    interpolation=cv2.INTER_AREA,
                )
                X_test = resized[np.newaxis, :, :, :]

                """Predict"""
                model = tf.keras.models.load_model(
                    os.path.join(models_path, f"{model_name}.h5")
                )
                y_pred = model.predict(X_test)
                print(y_pred)
                x, y = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
                start_point_1, end_point_1 = (round(x * y_pred[0, 0]), 0), (
                    round(x * y_pred[0, 0]),
                    y,
                )
                start_point_2, end_point_2 = (round(x * y_pred[0, 1]), 0), (
                    round(x * y_pred[0, 1]),
                    y,
                )

                # Green color in BGR
                color = (0, 255, 0)

                # Line thickness of 9 px
                thickness = 1

                # # Using cv2.line() method
                # # Draw a diagonal green line with thickness of 9 px
                image = cv2.line(resized, start_point_1, end_point_1, color, thickness)
                image = cv2.line(resized, start_point_2, end_point_2, color, thickness)

                cv2.imwrite(os.path.join(inference_path, name), resized)

                k += 1

            # Or multiple images
            wandb.log(
                {
                    "example": [
                        wandb.Image(img)
                        for img in glob(os.path.join(inference_path, "*.png"))
                    ]
                }
            )

            wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
