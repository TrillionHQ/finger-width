# -*- coding: utf-8 -*-
import os
import wandb
import click
import config
import logging
import numpy as np


from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


@click.command()
@click.argument(
    "processed_path", default=config.PROCESSED_PATH, type=click.Path(exists=True)
)
@click.argument("interim_path", default=config.INTERIM_PATH, type=click.Path())
@click.argument("external_path", default=config.EXTERNAL_PATH, type=click.Path())
@click.option("--data", default=config.DATA_NAME, help="type of data")
def main(processed_path: str, interim_path: str, external_path: str, data: str) -> None:
    """
    Split dataset
    :param processed_path:
    :param interim_path:
    :param external_path:
    :param data:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Split dataset")

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="Perceptron-finger",
        job_type="split_data",
        tags=[f"{data}"],
    ) as run:
        # download dataset from artifact
        artifact = run.use_artifact(f"{data}:latest", type="dataset")
        if "real" in data:
            dataset_dir = artifact.download(interim_path)
            dataset = np.load(os.path.join(interim_path, f"{data}_crop_finger.npz"))
        elif "synthetic" in data:
            dataset_dir = artifact.download(external_path)
            dataset = np.load(os.path.join(external_path, f"{data}_crop_finger.npz"))
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["x"], dataset["y"], test_size=config.TEST_SPLIT, random_state=42
        )

        # save splited dataset
        np.save(os.path.join(processed_path, f"X_{data}_train.npy"), X_train)
        np.save(os.path.join(processed_path, f"X_{data}_test.npy"), X_test)
        np.save(os.path.join(processed_path, f"y_{data}_train.npy"), y_train)
        np.save(os.path.join(processed_path, f"y_{data}_test.npy"), y_test)

        # Create a new artifact for the syntetic data, including the function that created it, to Artifacts
        ds_art = wandb.Artifact(
            name=f"splited_{data}",
            type="dataset",
            description="Split dataset into random train and test subsets ",
            metadata={
                "X_train_shape": X_train.shape,
                "y_train_shape": y_train.shape,
                "X_test_shape": X_test.shape,
                "y_test_shape": y_test.shape,
            },
        )

        # Attach our processed data to the Artifact
        ds_art.add_file(os.path.join(processed_path, f"X_{data}_train.npy"))
        ds_art.add_file(os.path.join(processed_path, f"X_{data}_test.npy"))
        ds_art.add_file(os.path.join(processed_path, f"y_{data}_train.npy"))
        ds_art.add_file(os.path.join(processed_path, f"y_{data}_test.npy"))

        # Log the Artifact
        run.log_artifact(ds_art)

        wandb.finish


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
