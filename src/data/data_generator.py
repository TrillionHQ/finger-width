# -*- coding: utf-8 -*-
import os
import cv2
import json
from src.data import config
import logging
import numpy as np
import tensorflow as tf

from glob import glob
from sklearn.utils import shuffle


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
    x = cv2.resize(
        x, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
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
    x.set_shape([config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    y.set_shape(
        [
            2,
        ]
    )

    return x, y


def tf_dataset(X, Y, batch):
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.map(tf_parse)
    data = data.batch(batch)
    data = data.prefetch(10)
    return data


def dataset(
    raw_path,
    batch_size,
    type_data,
):
    logger = logging.getLogger(__name__)
    logger.info(f"{type_data.upper()} data generation")

    train_x, train_y = load_data(raw_path)
    train_x, train_y = shuffling(train_x, train_y)

    print(f"[INFO] {type_data.upper()} data generator dataset created {len(train_x)} photos with {config.IMAGE_SIZE} resolution")

    data = tf_dataset(train_x, train_y, batch=batch_size)

    return data, len(train_x)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_dataset = dataset(config.RAW_PATH, 16, "train")

    # read dataset
    dataset = train_dataset[0].map(lambda x, y: x)  # read x dataset
    # dataset = train_dataset.map(lambda x, y: y)  # read x dataset
    for batch in dataset.as_numpy_iterator():
        print(batch.shape)
        # print("Batch is..")
        # print(batch)  # read first batch
        break
