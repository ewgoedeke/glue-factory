"""
Run the homography adaptation with Superpoint for all images in the minidepth dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse

import numpy as np
import torch
import time

from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.settings import EVAL_PATH
from gluefactory.datasets import get_dataset
from gluefactory.models.extractors.superpoint_open import SuperPoint
from omegaconf import OmegaConf

from gluefactory.models import get_model

model_configs = {
    "aliked": {
        "name": 'extractors.aliked',
        "trainable": False,
        "max_num_keypoints": 1024,
        "nms_radius": 4,
        "detection_threshold": -1,
        "trainings": {'do': False},
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,  # local path of pretrained weights
    },

    "sp": {
        "name": "extractors.superpoint_open",
        "max_num_keypoints": 1024,
        "nms_radius": 4,
        "detection_threshold": -1,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,  # local path of pretrained weights
    },

    "deeplsd": {
        "name": "lines.deeplsd",
        "min_length": 25,
        "max_num_lines": None,
        "force_num_lines": False,
        "model_conf": {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3.0,
            },
            "optimize": False,
        },
        "optimize": False,
    },

    "sold2": {
        "name": "lines.sold2",
        "min_length": 25,
        "max_num_lines": None,
        "force_num_lines": False,
        "model_conf": {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3.0,
            },
        }
    },

    "jpldd": {
        "name": "extractors.jpldd.joint_point_line_extractor",
        "aliked_model_name": "aliked-n16",
        "max_num_keypoints": 1024,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "force_num_keypoints": False,
        "training": {  # training settings
            "do": False,
        },
        "line_detection": {
            "do": True,
        },
        # "checkpoint": "../../../teamfolder/outputs/training/rk_jpldd_08_correct_gt_step3/checkpoint_best.tar",
        "checkpoint": "../../../teamfolder/outputs/training/rk_jpldd_09_pretrained_faster/checkpoint_best.tar",
        "nms_radius": 2,
        "line_neighborhood": 5,  # used to normalize / denormalize line distance field
        "timeit": False,  # override timeit: False from BaseModel
        # "line_df_decoder_channels": 64,
        # "line_af_decoder_channels": 64,
    }
}


def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': False,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'train_batch_size': 1,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'test'  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'test'))
    return loader


def run_measurement(dataloader, model, num_s, name):
    count = 0
    timings = []
    for img in tqdm(dataloader):
        if torch.cuda.is_available():
            # scale = img['image'].new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            # gs_image = (img['image'] * scale).sum(1, keepdim=True)
            # img = gs_image.to('cuda')
            img['image'] = img['image'].to('cuda')
            torch.cuda.synchronize()

        with torch.no_grad():
            start = time.time()
            pred = model(img)
            end = time.time()
            timings.append((end - start))

        count += 1

        if count == num_s:
            break
    print(f"*** RESULTS FOR {name} ON {num_s} IMAGES ***")
    print(f"\tMean: {np.mean(timings)}")
    print(f"\tMedian: {np.median(timings)}")
    print(f"\tMax: {np.max(timings)}")
    print(f"\tMin: {np.min(timings)}")
    print(f"\tStd: {np.std(timings)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, choices=['aliked', 'sp', 'deeplsd', 'jpldd'])
    parser.add_argument('--num_s', type=int, default=100, help='Number of timing samples.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1,
                        help='Number of jobs the dataloader uses to load images')
    args = parser.parse_args()

    print("NUMBER OF SAMPLES: ", args.num_s)
    print("MODEL TO TEST: ", args.config)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = model_configs[args.config]
    model_name = config["name"]
    model = get_model(model_name)(config)
    model.eval().to(device)

    run_measurement(dataloader, model, args.num_s, model_name)


