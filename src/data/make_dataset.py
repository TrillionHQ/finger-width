# -*- coding: utf-8 -*-
import os
import cv2
import json
import click
import wandb
import config
import logging
import numpy as np

from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()
@click.command()
@click.argument("raw_path", default=config.RAW_PATH, type=click.Path(exists=True))
@click.argument(
    "interim_path", default=config.INTERIM_PATH, type=click.Path(exists=True)
)
@click.option("--height", default=config.IMAGE_SIZE[0], help="height")
@click.option("--width", default=config.IMAGE_SIZE[1], help="width")
@click.option("--channels", default=config.IMAGE_SIZE[2], help="number of channels")
@click.option("--data", default=config.DATA_NAME, help="type of data")
@click.option("--entity", default=os.getenv("WANDB_ENTITY"), help="entity")
@click.option("--project", default=os.getenv("WANDB_PROJECT"), help="project")
def main(
    raw_path: str, interim_path: str, height: str, width: str, channels: str, data: str, entity: str, project: str
) -> None:
    """
    Runs real data preprocessing scripts and saved it to ../interim_path).
    :param raw_path:
    :param interim_path:
    :param data:
    :param height:
    :param width:
    :param channels:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Real data generation")

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        entity=entity,
        project=project,
        job_type="preprocessed-data",
        tags=[f"{data}"],
    ) as run:
        images = sorted(glob(os.path.join(raw_path, "*.png")))
        js = sorted(glob(os.path.join(raw_path, "*.json")))

        x = np.ndarray(shape=(len(images), height, width, channels), dtype=np.float32)
        y = np.ndarray(shape=(len(images), 2), dtype=np.float32)

        i = 0
        for k, j in tqdm(zip(images, js), total=len(images)):
            if "fl" in data:
                img = cv2.imread(k).astype(np.float32)/255.0
            else:
                img = cv2.imread(k)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image
            resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            with open(j, "r") as f:
                points = json.load(f)

            x[i] = resized
            y[i] = np.array([points["left"], points["right"]])
            i += 1

        # write placeholder arrays into a binary npz file
        np.savez(
            os.path.join(interim_path, f"{data}_crop_finger.npz"), x=x, y=y
        )

        # Create a new artifact for the syntetic data, including the function that created it, to Artifacts
        ds_art = wandb.Artifact(
            name=f"{data}",
            type="dataset",
            description="Processed dataset into uncompressed .npz format",
            metadata={"X_shape": x.shape, "y_shape": y.shape},
        )

        # Attach our processed data to the Artifact
        ds_art.add_file(
            os.path.join(interim_path, f"{data}_crop_finger.npz")
        )

        # Log the Artifact
        run.log_artifact(ds_art)

        wandb.finish

        print(f"Real dataset created ({i} photos)")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
