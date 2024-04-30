# Test model loading and initialization

import argparse
import time
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..models import get_model
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions

logger = logging.getLogger(__name__)

jpldd_keys = ["keypoints",
              "keypoint_descriptors",
              "keypoint_scores",
              "keypoint_score_dispersity",
              "keypoint_and_junction_score_map",
              "deeplsd_line_anglefield",
              "deeplsd_line_distancefield",
              #"line_endpoints",
              #"line_descriptors"
              ]


def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': False,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'test_batch_size': 2,
        'train_batch_size': 2,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'test',  # if implemented by dataset class gives different splits
        'prefetch_factor': None,
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'train'), pinned=True, distributed=False)
    return dataset, loader


def get_model_object(model_name):
    model_specific_config = {  # define model specific config
        'pretrained': True
    }
    omega_ms_conf = OmegaConf.create(model_specific_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name)(omega_ms_conf).eval().to(device)
    return model


def run(feature_file, model_name, num_workers_dataloader):
    print("Load dataset and create dataloader...")
    dataset, data_loader = get_dataset_and_loader(num_workers=num_workers_dataloader)
    print(f"Load model: {model_name}")
    model = get_model_object(model_name)
    export_predictions(data_loader, model, feature_file, as_half=True, keys=jpldd_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)  # model to load and export features with
    parser.add_argument("--export_name", type=str, default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    logger.info("Running Export...")
    # export results to one h5 file
    feature_file = Path(DATA_PATH, "exports", args.export_name + ".h5")
    feature_file.parent.mkdir(exist_ok=True, parents=True)
    logger.info(
        f"Export local features for dataset minidepth "
        f"to file {feature_file}"
    )
    run(feature_file, args.model, args.num_workers)
    logger.info("Done!")
