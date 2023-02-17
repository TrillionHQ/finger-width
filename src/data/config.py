# import the necessary packages
import os

# initialize the path to the *original* input directory of images
RAW_PATH = "../../data/raw"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
INTERIM_PATH = "../../data/interim"

# The final, canonical data sets for modeling
PROCESSED_PATH = "../../data/processed"

EXTERNAL_PATH = "../../data/external"

TEST_PATH = "../../data/test"

INFERENCE_PATH = "../../data/inference"

MODELS_PATH = "../../models"

DATA_TYPE = "real"  # synthetic
DATA_DTYPE = ""
# define the amount of data that will be used training
TEST_SPLIT = 0.25

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.15

# define input image spatial dimensions
IMAGE_SIZE = (20, 50, 3)

DATA_NAME = f"{DATA_TYPE}_{DATA_DTYPE}_{IMAGE_SIZE[0]}_{IMAGE_SIZE[1]}"

default_config = dict(
    img_size=40,
    batch_size=16,
    epochs=10,
    lr=0.001,
    val_split=0.15,
    earlystop=20,
    reduce_lr=5,
    arch="Perceptron",
    size_layer1=256,
    size_layer2=32,
    seed=42,
)
