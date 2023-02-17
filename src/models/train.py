# -*- coding: utf-8 -*-
import os
import wandb
import click
import logging
import numpy as np
import tensorflow as tf

from typing import Tuple
from metrics import r2_score
from dotenv import load_dotenv
from src.data import config
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from wandb.keras import WandbMetricsLogger

default_config = dict(
    img_size=40,
    batch_size=16,
    epochs=10,
    lr=0.001,
    val_split=0.15,
    earlystop=20,
    reduce_lr=5,
    arch='Perceptron',
    size_layer1=256,
    size_layer2=32,
    data='real',
    seed=42,
    )

@click.command()
@click.argument("processed_path", default=config.PROCESSED_PATH, type=click.Path(exists=True))
@click.argument("models_path", default=config.MODELS_PATH, type=click.Path())
@click.option('--img_size', default=default_config['img_size'], help='image size')
@click.option('--batch_size', default=default_config['batch_size'], help='batch size')
@click.option('--epochs', default=default_config['epochs'], help='number of training epochs')
@click.option('--lr', default=default_config['lr'], help='learning rate')
@click.option('--val_split', default=default_config['val_split'], help='validation_split')
@click.option('--earlystop', default=default_config['earlystop'], help='earlystopping_patience')
@click.option('--reduce_lr', default=default_config['reduce_lr'], help='reduce_LR_onp_lateau_patience')
@click.option('--arch', default=default_config['arch'], help='architecture')
@click.option('--size_layer1', default=default_config['size_layer1'], help='number of neurons in hidden layer')
@click.option('--size_layer2', default=default_config['size_layer2'], help='number of neurons in hidden layer')
@click.option('--seed', default=default_config['seed'], help='random seed')
@click.option('--data', default=default_config['data'], help='real or synthetic')
def main(processed_path: str, models_path: str, img_size: int, batch_size: int , epochs: int, lr: float, val_split: float, earlystop: int, reduce_lr: int, arch: str, size_layer1: int, size_layer2: int, seed: int, data: str) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    :param processed_path:
    :param models_path:
    :param height:
    :param width:
    :param channels:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model")

    load_dotenv()
    wandb.login(key=os.getenv('WANDB_API_KEY'))

    with wandb.init(entity=os.getenv('WANDB_ENTITY'), project='Perceptron-finger', job_type='train', tags=[f'{data}'],
                    config=default_config) as run:  # config is optional here

        # download dataset from artifact
        artifact = run.use_artifact(f'splited_{data}_data:latest', type='dataset')
        artifact_dir = artifact.download(processed_path)

        model_path = os.path.join(models_path, f'{data}_model.h5')
        csv_path = os.path.join(models_path, f'{data}_logger.csv')


        """Model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(img_size, img_size, 3))) # flatten the input
        model.add(tf.keras.layers.Dense(units=size_layer1, activation='relu')) # hidden layer
        model.add(tf.keras.layers.Dense(units=size_layer2, activation='relu'))
        model.add(tf.keras.layers.Dense(units=2)) # output layer with 2 units and softmax activation

        # Compile the model
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), r2_score])

        # Print a summary of the model's architecture
        model.summary()

        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_lr, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=earlystop, restore_best_weights=False)
        ]

        hist = model.fit(
            x=np.load(os.path.join(processed_path, f'X_{data}_train.npy')),
            y=np.load(os.path.join(processed_path, f'y_{data}_train.npy')),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[callbacks, WandbMetricsLogger()],
            verbose=1)

        # write summary
        run.summary['best_loss'] = max(hist.history['loss'])
        run.summary['best_val_loss'] = max(hist.history['val_loss'])
        run.summary['best_r2_score'] = max(hist.history['r2_score'])
        run.summary['best_val_r2_score'] = max(hist.history['val_r2_score'])
        run.summary['best_mean_squared_error'] = max(hist.history['mean_squared_error'])
        run.summary['best_val_mean_squared_error'] = max(hist.history['val_mean_squared_error'])


        """Save the Trained Model to W&B Artifacts"""
        model.save(model_path)
        # Log the trained model to W&B Artifacts, including the config
        model_art = wandb.Artifact(name=f'{data}_model', type='model') #metadata=dict(hyper_param)
        model_art.add_file(model_path)
        run.log_artifact(model_art)

        """Predict"""
        # X_test = np.load(os.path.join(processed_path, f'X_{data}_test.npy'))
        # y_test = np.load(os.path.join(processed_path, f'y_{data}_test.npy'))
        #
        # with CustomObjectScope({'mean_absolute_error': mean_absolute_error, 'mean_squared_error': mean_squared_error, 'r2_score': r2_score}):
        #     model = tf.keras.models.load_model(os.path.join(models_path, f'{data}_model.h5'))
        #     y_pred = model.predict(X_test)
        #
        #     score = dict(
        #         r2=round(r2_score(K.constant(y_test), K.constant(y_pred)).numpy(), 2),
        #         mse=round(mean_squared_error(y_test, y_pred), 2),
        #         mae=round(mean_absolute_error(y_test, y_pred), 2)
        #     )
        #
        #     print(f'Coefficient of determination of model is {score["r2"]}')
        #     print(f'MSE metric of model is {score["mse"]}')
        #     print(f'MAE metric of model is {score["mae"]}')
        #
        #     wandb.log({'mean_absolute_error': score["mae"], 'mean_squared_error': score["mse"], 'r2_score': score["r2"]})

        wandb.finish()

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
