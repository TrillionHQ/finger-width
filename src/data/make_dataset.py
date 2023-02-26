import config
import shutil
import os
from glob import glob
from sklearn.model_selection import train_test_split


def create_dir(path):
    """Creating a directory"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_data(path, split=0.1):
    """Loading the images and masks"""
    X = sorted(glob(os.path.join(path, "*.png")))
    Y = sorted(glob(os.path.join(path, "*.json")))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    train_x, val_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, val_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y), (val_x, val_y)


def save_data(path, splited_data):
    folders = ["train", "evaluate", "valid"]
    for i, d in enumerate(folders):
        dst_dir = create_dir(os.path.join(path, d))

        for im, js in zip(splited_data[i][0], splited_data[i][1]):
            dst_im = os.path.join(dst_dir, im.split("/")[-1])
            dst_js = os.path.join(dst_dir, js.split("/")[-1])
            if os.path.isfile(dst_im):
                pass
            else:
                shutil.copy(im, dst_im)
                shutil.copy(js, dst_js)
    print(f"Dataset splitted into {path}")
    print(
        f"Train: {round(len(os.listdir(os.path.join(path, 'train')))/2)}, "
        f"Evaluate: {round(len(os.listdir(os.path.join(path, 'evaluate')))/2)}, "
        f"Valid: {round(len(os.listdir(os.path.join(path, 'valid')))/2)}"
    )


save_data(config.PROCESSED_PATH, load_data(config.RAW_PATH))
