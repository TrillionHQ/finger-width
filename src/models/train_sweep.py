# -*- coding: utf-8 -*-
import os
import cv2
import wandb
import click
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from glob import glob
from keras import metrics
from src.data import config
from metrics import r2_score
from dotenv import load_dotenv
from wandb.keras import WandbMetricsLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)


# Sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        # "img_size": {"value": 40},
        "batch_size": {"value": 64},
        "epochs": {"value": 1000},
        "lr": {"value": 1e-4},
        "val_split": {"value": 0.15},
        "earlystop": {"value": 20},
        "reduce_lr": {"value": 5},
        "arch": {"value": "Perceptron"},
        "size_layer1": {"values": [64, 128, 256, 512]},
        "size_layer2": {"values": [16, 32, 64, 128, 256]},
        "seed": {"value": 42},
        "pred_del": {"value": 100},
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="Finger_width")


@click.command()
@click.argument(
    "processed_path", default=config.PROCESSED_PATH, type=click.Path(exists=True)
)
@click.argument("models_path", default=config.MODELS_PATH, type=click.Path())
@click.argument("test_path", default=config.TEST_PATH, type=click.Path())
@click.argument("inference_path", default=config.INFERENCE_PATH, type=click.Path())
@click.option("--data", default=config.DATA_NAME, help="type of data")
def main(
    processed_path: str,
    models_path: str,
    test_path: str,
    inference_path: str,
    data: str,
) -> None:
    """
    Hyperparameter search and model optimization
    :param processed_path
    :param models_path
    :param data
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Hyperparameter search and model optimization")

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="Finger_width",
        job_type="sweep",
        tags=[f"{data}"],
    ) as run:  # config is optional here
        # get sweep parameters
        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs
        lr = wandb.config.lr
        val_split = wandb.config.val_split
        earlystop = wandb.config.earlystop
        reduce_lr = wandb.config.reduce_lr
        arch = wandb.config.arch
        size_layer1 = wandb.config.size_layer1
        size_layer2 = wandb.config.size_layer2
        seed = wandb.config.seed
        pred_del = wandb.config.pred_del

        # download dataset from artifact
        artifact = run.use_artifact(f"split_{data}:latest", type="dataset")
        artifact_dir = artifact.download(processed_path)
        X_train = np.load(os.path.join(processed_path, f"X_{data}_train.npy"))
        y_train = np.load(os.path.join(processed_path, f"y_{data}_train.npy"))
        y_train = y_train * pred_del

        model_path = os.path.join(models_path, f"{data}.h5")
        csv_path = os.path.join(models_path, f"{data}_logger.csv")

        """Model"""
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Flatten(
                input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
            )
        )  # flatten the input
        model.add(
            tf.keras.layers.Dense(units=size_layer1, activation="relu")
        )  # hidden layer
        model.add(tf.keras.layers.Dense(units=size_layer2, activation="relu"))
        model.add(
            tf.keras.layers.Dense(units=2)
        )  # output layer with 2 units and softmax activation

        # Compile the model
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=opt,
            loss="mean_squared_error",
            metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, r2_score],
        )

        # Print a summary of the model's architecture
        model.summary()

        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=reduce_lr,
                min_lr=1e-7,
                verbose=1,
            ),
            CSVLogger(csv_path),
            TensorBoard(),
            EarlyStopping(
                monitor="val_loss", patience=earlystop, restore_best_weights=False
            ),
        ]

        hist = model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[callbacks, WandbMetricsLogger()],
            verbose=1,
        )

        # write summary
        run.summary["best_loss"] = min(hist.history["loss"])
        run.summary["best_val_loss"] = min(hist.history["val_loss"])
        run.summary["best_r2_score"] = max(hist.history["r2_score"])
        run.summary["best_val_r2_score"] = max(hist.history["val_r2_score"])
        run.summary["best_mean_squared_error"] = min(hist.history["mean_squared_error"])
        run.summary["best_val_mean_squared_error"] = min(
            hist.history["val_mean_squared_error"]
        )

        """Save the Trained Model to W&B Artifacts"""
        model.save(model_path)
        model_size = round(os.stat(model_path).st_size / 1000000, 1)
        # Log the trained model to W&B Artifacts, including the config
        model_art = wandb.Artifact(
            name=f"{data}_model",
            type="model",
            metadata=dict(config.default_config),
        )
        model_art.add_file(model_path)
        run.log_artifact(model_art)

        """Predict"""
        with CustomObjectScope(
            {
                "mean_absolute_error": metrics.mean_absolute_error,
                "mean_squared_error": metrics.mean_squared_error,
                "r2_score": r2_score,
            }
        ):
            images = sorted(glob(os.path.join(test_path, "*.png")))
            js = sorted(glob(os.path.join(test_path, "*.json")))

            k = 1
            points = {"left": [], "right": []}
            pred_data = []
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
                model = tf.keras.models.load_model(model_path)
                y_pred = model.predict(X_test)
                y_pred = y_pred / pred_del
                points["left"].append(y_pred[0, 0]), points["right"].append(
                    y_pred[0, 1]
                )
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

                pred_data.append(
                    [
                        k,
                        wandb.Image(os.path.join(inference_path, name), resized),
                        name,
                        y_pred[0, 0],
                        y_pred[0, 1],
                    ]
                )

                k += 1

        # # Or multiple images
        # inference_images = glob(os.path.join(inference_path, "*.png"))
        # wandb.log(
        #     {
        #         "example": [
        #             wandb.Image(img)
        #             for img in inference_images
        #         ]
        #     }
        # )

        # create a wandb.Table() with corresponding columns
        columns = ["id", "image", "name", "left", "right"]
        table = wandb.Table(data=pred_data, columns=columns)

        wandb.log({"Table": table})

        wandb.log(
            {"num_images": X_train.shape[0] * (1 - val_split), "model_size": model_size}
        )

        wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=40)
