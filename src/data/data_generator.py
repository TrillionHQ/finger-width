# -*- coding: utf-8 -*-
import os
import cv2
import json
import logging
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.utils import shuffle


load_dotenv()


def dataset(
    raw_path: str,
    height: str,
    width: str,
    channels: str,
    batch_size: int,
) -> None:
    """
    Runs real data preprocessing scripts and saved it to ../interim_path).
    :param raw_path:
    :param height:
    :param width:
    :param channels:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Real data generation")

    def shuffling(x, y):
        x, y = shuffle(x, y, random_state=42)
        return x, y

    def load_data(path):
        x = sorted(glob(os.path.join(path, "*.png")))
        y = sorted(glob(os.path.join(path, "*.json")))
        return x, y

    def read_image(path):
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA)
        x = x / 255.0
        x = x.astype(np.float32)
        return x

    def read_json(path):
        path = path.decode()
        with open(path, "r") as f:
            points = json.load(f)
        y = np.array([points["left"], points["right"]], dtype=np.float32)
        return y

    def tf_parse(x, y):
        def _parse(x, y):
            x = read_image(x)
            y = read_json(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([height, width, channels])
        y.set_shape(
            [
                2,
            ]
        )

        return x, y

    def tf_dataset(X, Y, batch):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map(tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(10)
        return dataset

    train_x, train_y = load_data(raw_path)
    train_x, train_y = shuffling(train_x, train_y)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Data generator dataset created ({len(train_x)} photos)")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)

    return train_dataset


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # train_dataset = dataset(
    #     config.RAW_PATH,
    #     224,
    #     224,
    #     3,
    #     16,
    # )

    # read dataset
    # dataset = train_dataset.map(lambda x, y: x) # read x dataset
    # #dataset = train_dataset.map(lambda x, y: y)  # read x dataset
    # for val in dataset.as_numpy_iterator():
    #     print(val.shape) # read first batch
    #     break
