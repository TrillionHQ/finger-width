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

TEST_PATH = "../../data/test/images"

INFERENCE_PATH = "../../data/test/inference"

MODELS_PATH = "../../models"

DATA_TYPE = "real"  # synthetic
LEN_DATA = len(glob(os.path.join(RAW_PATH, "*.png")))
# define the amount of data that will be used training
TEST_SPLIT = 0.15

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.15

# define input image spatial dimensions
IMAGE_SIZE = (224, 224, 3)

DATA_NAME = f"{DATA_TYPE}_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_{LEN_DATA}"
print(DATA_NAME)

default_config = dict(
    im_size=224,
    batch_size=16,
    epochs=200,
    lr=1e-4,
    earlystop=20,
    reduce_lr=5,
    arch="MobileNetV3_Small",
    size_layer1=256,
    size_layer2=32,
    seed=42,
    pred_del=10
)
