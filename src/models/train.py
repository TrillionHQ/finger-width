# -*- coding: utf-8 -*-
import os
import click
import logging

from keras import metrics
from metrics import r2_score
from src.data import config
from model import mobilenetv3_small
from src.data.data_generator import dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)

@click.command()
@click.argument("processed_path", default=config.PROCESSED_PATH, type=click.Path(exists=True))
@click.argument("models_path", default=config.MODELS_PATH, type=click.Path())
def train(processed_path: str, models_path: str) -> None:
    """
    Train model
    :param processed_path:
    :param models_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model")

    # get parameters
    IM_SIZE = config.default_config["im_size"]
    BS = config.default_config["batch_size"]
    EPOCHS = config.default_config["epochs"]
    DEFAULT_LR = config.default_config["lr"]
    EARLY_STOPPING_PATIENCE = config.default_config["earlystop"]
    REDUCE_LR_PATIENCE = config.default_config["reduce_lr"]
    ARCH = config.default_config["arch"]
    SIZE_LAYER_1 = config.default_config["size_layer1"]
    SIZE_LAYER_2 = config.default_config["size_layer2"]
    SEED = config.default_config["seed"]
    PRED_DEL = config.default_config["pred_del"]

    """Dataset"""

    train_dataset = dataset(
        os.path.join(processed_path, "train"), BS, "train"
    )
    test_dataset = dataset(
        os.path.join(processed_path, "test"), BS, "test"
    )
    valid_dataset = dataset(
        os.path.join(processed_path, "valid"), BS, "valid"
    )

    """Loggers"""
    model_path = os.path.join(models_path, f"{config.DATA_NAME}.h5")
    csv_path = os.path.join(models_path, f"{config.DATA_NAME}_logger.csv")

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
        callbacks=[callbacks],
        verbose=1,
    )

    loss_eval, mse_eval, mae_eval, r2_score_eval = model.evaluate(test_dataset[0])

    """Save the Trained Model to W&B Artifacts"""
    model.save(model_path)
    model_size = round(os.stat(model_path).st_size / 1000000, 1)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
