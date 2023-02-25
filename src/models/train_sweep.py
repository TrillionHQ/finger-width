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
from model import mobilenetv3_small
from src.data.data_generator import dataset
from src.data import config
from metrics import r2_score
from dotenv import load_dotenv
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope
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
        "im_size": {"value": config.default_config["im_size"]},
        "batch_size": {"value": config.default_config["batch_size"]},
        "epochs": {"value": config.default_config["epochs"]},
        "lr": {"value": config.default_config["lr"]},
        "earlystop": {"value": config.default_config["earlystop"]},
        "reduce_lr": {"value": config.default_config["reduce_lr"]},
        "arch": {"value": config.default_config["arch"]},
        "size_layer1": {"values": [16, 32, 64, 128]},
        "size_layer2": {"values": [8, 16, 32]},
        "seed": {"value": config.default_config["seed"]},
        "pred_del": {"value": 100},
    },
}


@click.command()
@click.argument(
    "processed_path", default=config.PROCESSED_PATH, type=click.Path(exists=True)
)
@click.argument("models_path", default=config.MODELS_PATH, type=click.Path())
@click.argument("test_path", default=config.TEST_PATH, type=click.Path())
@click.argument("inference_path", default=config.INFERENCE_PATH, type=click.Path())
@click.option("--data", default=config.DATA_NAME, help="data")
@click.option("--entity", default=os.getenv("WANDB_ENTITY"), help="entity")
@click.option("--project", default=os.getenv("WANDB_PROJECT"), help="project")
def main(
    processed_path: str,
    models_path: str,
    test_path: str,
    inference_path: str,
    data: str,
    entity: str,
    project: str,
) -> None:
    """
    Hyperparameter search and model optimization
    :param raw_path
    :param models_path
    :param data
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Hyperparameter search and model optimization")

    # model_
    model_name = f"{data}"

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=entity,
        project=project,
        job_type="sweep",
        tags=[f"{data}"],
    ) as run:
        # get sweep parameters
        IM_SIZE = wandb.config.im_size
        BS = wandb.config.batch_size
        EPOCHS = wandb.config.epochs
        DEFAULT_LR = wandb.config.lr
        EARLY_STOPPING_PATIENCE = wandb.config.earlystop
        REDUCE_LR_PATIENCE = wandb.config.reduce_lr
        ARCH = wandb.config.arch
        SIZE_LAYER_1 = wandb.config.size_layer1
        SIZE_LAYER_2 = wandb.config.size_layer2
        SEED = wandb.config.seed
        PRED_DEL = wandb.config.pred_del

        """Dataset"""

        train_dataset = dataset(os.path.join(processed_path, "train"), BS, "train")
        test_dataset = dataset(os.path.join(processed_path, "test"), BS, "test")
        valid_dataset = dataset(os.path.join(processed_path, "valid"), BS, "valid")

        """Loggers"""
        model_path = os.path.join(models_path, f"{data}.h5")
        csv_path = os.path.join(models_path, f"{data}_logger.csv")

        """Model"""
        model = mobilenetv3_small(IM_SIZE, IM_SIZE, 3)

        # Compile the model
        opt = Adam(learning_rate=DEFAULT_LR)
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
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1,
            ),
            CSVLogger(csv_path),
            TensorBoard(),
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=False,
            ),
        ]

        hist = model.fit(
            train_dataset[0],
            epochs=EPOCHS,
            batch_size=BS,
            validation_data=valid_dataset[0],
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
                    os.path.join(models_path, f"{data}.h5")
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
                resized = cv2.line(
                    resized, start_point_1, end_point_1, color, thickness
                )
                resized = cv2.line(
                    resized, start_point_2, end_point_2, color, thickness
                )

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

        loss_eval, mse_eval, mae_eval, r2_score_eval = model.evaluate(test_dataset[0])

        # create a wandb.Table() with corresponding columns
        columns = ["id", "image", "name", "left", "right"]
        table = wandb.Table(data=pred_data, columns=columns)

        wandb.log({"Table": table})

        wandb.log(
            {
                "loss_eval": round(loss_eval, 4),
                "mse_eval": round(mse_eval, 4),
                "mae_eval": round(mae_eval, 4),
                "r2_score_eval": round(r2_score_eval, 4),
                "num_images": train_dataset[1],
                "model_size": model_size,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Start sweep job.
    load_dotenv()
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
    )
    wandb.agent(sweep_id, function=main, count=20)
