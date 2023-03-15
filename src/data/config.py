# import the necessary packages
import os
from glob import glob

# initialize the path to the *original* input directory of images
RAW_PATH = "../../data/raw/images"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
INTERIM_PATH = "../../data/interim"

# The final, canonical data sets for modeling
PROCESSED_PATH = "../../data/processed"

EXTERNAL_PATH = "../../data/external"

TEST_PATH = "../../data/processed/test/images"

INFERENCE_PATH = "../../data/processed/test/inference"

MODELS_PATH = "../../models"

DATA_TYPE = "real"  # synthetic
LEN_DATA = len(glob(os.path.join(RAW_PATH, "*.png")))
# define the amount of data that will be used training
TEST_SPLIT = 0.15

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.15

# define input image spatial dimensions
IMAGE_SIZE = 224

DATA_NAME = f"{DATA_TYPE}_{IMAGE_SIZE}_{LEN_DATA}"
print(DATA_NAME)

default_config = dict(
    im_size=IMAGE_SIZE,
    batch_size=32,
    epochs=200,
    lr=1e-4,
    early_stop=10,
    reduce_lr=5,
    arch="MobileNetV3_Small",
    alpha=0.75,
    size_layer1=64,
    seed=42,
    pred_del=10
)
