import os
import cv2
import json
import shutil
import logging
import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Any
from src.data import config
from sklearn.model_selection import train_test_split


def create_dir(path):
    """Creating a directory"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def image_resized(path: str, im_size: int) -> tuple[Any, Any]:
    """
    Resize and normalize image
    :param path:
    :param im_size:
    :return:
    """
    x = cv2.imread(path)
    resized = cv2.resize(
        x,
        (im_size, im_size),
        interpolation=cv2.INTER_AREA,
    )
    x = resized / 255.0
    x = x.astype(np.float32)

    x = x[np.newaxis, :, :, :]

    return x, resized


def check_equality_data(lst1: list[str], lst2: list[str]) -> list[str]:
    """
    Check folders for intersection
    :param lst1:
    :param lst2:
    :return:
    """
    print(
        f"[INFO] Image data and json data before clean: images={len(lst1)} and json={len(lst2)}"
    )
    names1 = [i.split("/")[-1].split(".")[0] for i in lst1]
    names2 = [j.split("/")[-1].split(".")[0] for j in lst2]

    intersection = set(names1) & set(names2)
    symmetric_difference = set(names1) ^ set(names2)
    if len(symmetric_difference) > 0:
        print(
            f"[INFO] Image data and json data have symmetric_difference ({len(symmetric_difference)} files)"
        )
    else:
        print(
            f"[INFO] Image data and json doesn't have symmetric_difference: images={len(names1)} and json={len(names2)}"
        )
    return list(symmetric_difference)


def clean_data(
    lst1: list[str], lst2: list[str], src_dst: str
) -> tuple[list[str], list[str]]:
    """
    If file hasn't pair, it deletes
    :param lst1:
    :param lst2:
    :param src_dst:
    :return:
    """
    diff_list = check_equality_data(lst1, lst2)
    if len(diff_list) > 0:
        for file_name in diff_list:
            path_im = os.path.join(src_dst, f"{file_name}.png")
            path_js = os.path.join(src_dst, f"{file_name}.json")
            if os.path.exists(path_im):
                os.remove(path_im)
                lst1.remove(path_im)
                print(f"{path_im} deleted")
            else:
                os.remove(path_js)
                lst2.remove(path_js)
                print(f"{path_js} deleted")
        print(f"[INFO] After data clean images={len(lst1)} and json={len(lst2)}")
    return lst1, lst2


def draw_line(
    im_size: int, name: str, left: int, right: int, resized: np.ndarray, dst: str
) -> np.ndarray:
    """
    Draw lines on image
    :param im_size:
    :param name:
    :param left:
    :param right:
    :param resized:
    :param dst:
    :return:
    """
    start_point_1, end_point_1 = (round(im_size * left), 0), (
        round(im_size * left),
        im_size,
    )
    start_point_2, end_point_2 = (round(im_size * right), 0), (
        round(im_size * right),
        im_size,
    )

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 1

    # # Using cv2.line() method
    # # Draw a diagonal green line with thickness of 9 px
    resized = cv2.line(resized, start_point_1, end_point_1, color, thickness)
    resized = cv2.line(resized, start_point_2, end_point_2, color, thickness)

    cv2.imwrite(os.path.join(dst, name), resized)

    return resized


def parse_data(
    lst1: list[str], lst2: list[str], src: str, dst: str, im_size: int
) -> None:
    """
    Function for labeling checking. Parse through folders, draw lines and save images to data/external
    :param im_size:
    :param src:
    :param dst:
    :param lst1:
    :param lst2:
    :return:
    """
    im, js = clean_data(lst1, lst2, src)
    for i, j in tqdm(zip(im, js), total=len(im)):
        # read image
        name = i.split("/")[-1]
        x = cv2.imread(i)
        resized = cv2.resize(
            x,
            (im_size, im_size),
            interpolation=cv2.INTER_AREA,
        )

        with open(j, "r") as f:
            points = json.load(f)

        draw_line(im_size, name, points["left"], points["right"], resized, dst)

    print("[INFO] Drew borders on all images")


def load_data(
    src: str, dst: str, im_size: int, split=0.1
) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]:
    """
    Loading the images and masks
    :param src:
    :param dst:
    :param im_size:
    :param split:
    :return:
    """
    X = sorted(glob(os.path.join(src, "*.png")))
    Y = sorted(glob(os.path.join(src, "*.json")))

    parse_data(X, Y, src, dst, im_size)

    """ Splitting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    train_x, val_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, val_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y), (val_x, val_y)


def split_data(path: str, loaded_data: tuple[tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]):
    """
    Split dataset to train/valid/evaluate
    :param path:
    :param loaded_data:
    :return:
    """
    folders = ["train", "evaluate", "valid"]
    for i, d in enumerate(folders):
        dst_dir = create_dir(os.path.join(path, d))

        for im, js in zip(loaded_data[i][0], loaded_data[i][1]):
            dst_im = os.path.join(dst_dir, im.split("/")[-1])
            dst_js = os.path.join(dst_dir, js.split("/")[-1])
            if os.path.isfile(dst_im):
                pass
            else:
                shutil.copy(im, dst_im)
                shutil.copy(js, dst_js)
    print(f"[INFO] Dataset split into {path}")
    print(
        f"Train: {round(len(os.listdir(os.path.join(path, 'train'))) / 2)}, "
        f"Evaluate: {round(len(os.listdir(os.path.join(path, 'evaluate'))) / 2)}, "
        f"Valid: {round(len(os.listdir(os.path.join(path, 'valid'))) / 2)}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    split_data(path=config.PROCESSED_PATH, loaded_data=load_data(config.RAW_PATH, config.EXTERNAL_PATH, config.IMAGE_SIZE))

