# -*- coding: utf-8 -*-
import os
import cv2
import wandb
import click
import logging
import tensorflow as tf

from tqdm import tqdm
from glob import glob
from keras import metrics
from model import create_mobilevit
from src.data.data_generator import dataset
from src.data.make_dataset import draw_line, image_resized
from src.data import config
from metrics import r2_score
from dotenv import load_dotenv
from wandb.keras import WandbMetricsLogger
from keras.optimizers import Adam
from keras.utils.layer_utils import count_params
from keras.utils import CustomObjectScope
from keras.callbacks import (
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
        "epochs": {"values": [5, 10]},
        "lr": {"value": config.default_config["lr"]},
        "early_stop": {"value": config.default_config["early_stop"]},
        "reduce_lr": {"value": config.default_config["reduce_lr"]},
        "arch": {"value": config.default_config["arch"]},
        "alpha": {"value": config.default_config["alpha"]},
        "seed": {"value": config.default_config["seed"]},
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
    :param processed_path:
    :param models_path:
    :param test_path:
    :param inference_path:
    :param data:
    :param entity:
    :param project:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Hyperparameter search and model optimization")

    # model
    model_name = f"{data}"

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=entity,
        project=project,
        job_type="sweep",
        tags=[f"{data}"],
    ) as run:

        """Data"""

        train_dataset = dataset(
            os.path.join(processed_path, "train"), wandb.config.batch_size, "train"
        )
        evaluate_dataset = dataset(
            os.path.join(processed_path, "evaluate"),
            wandb.config.batch_size,
            "evaluate",
        )
        valid_dataset = dataset(
            os.path.join(processed_path, "valid"), wandb.config.batch_size, "valid"
        )

        """Loggers"""
        model_path = os.path.join(models_path, f"{data}.h5")
        csv_path = os.path.join(models_path, f"{data}_logger.csv")

        """Model"""
        model = create_mobilevit(im_size=wandb.config.im_size, channels=3, num_outputs=2)

        # Compile the model
        opt = Adam(learning_rate=wandb.config.lr)
        model.compile(
            optimizer=opt,
            loss="mean_squared_error",
            metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, r2_score],
        )

        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=wandb.config.reduce_lr,
                min_lr=1e-7,
                verbose=1,
            ),
            CSVLogger(csv_path),
            TensorBoard(),
            EarlyStopping(
                monitor="val_loss",
                patience=wandb.config.early_stop,
                restore_best_weights=True,
            ),
        ]

        hist = model.fit(
            train_dataset[0],
            epochs=wandb.config.epochs,
            batch_size=wandb.config.batch_size,
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

                x, resized = image_resized(i, wandb.config.im_size)

                """Predict"""
                model = tf.keras.models.load_model(
                    os.path.join(models_path, f"{data}.h5")
                )
                y_pred = model.predict(x)
                print(y_pred)

                pred_image = draw_line(
                    wandb.config.im_size,
                    name,
                    y_pred[0, 0],
                    y_pred[0, 1],
                    resized,
                    inference_path,
                )

                pred_data.append(
                    [
                        k,
                        wandb.Image(os.path.join(inference_path, name), pred_image),
                        name,
                        y_pred[0, 0],
                        y_pred[0, 1],
                    ]
                )

                k += 1

        loss_eval, mse_eval, mae_eval, r2_score_eval = model.evaluate(
            evaluate_dataset[0]
        )

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
                "trainable_params": count_params(model.trainable_weights),
                "non_trainable_params": count_params(model.non_trainable_weights),
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
    wandb.agent(sweep_id, function=main, count=3)
