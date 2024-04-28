"""
Run the homography adaptation with Superpoint for all images in the minidepth dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import h5py
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed

from gluefactory.settings import EVAL_PATH
from gluefactory.datasets import get_dataset
from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.ground_truth_generation.generate_gt_deeplsd import generate_ground_truth_with_homography_adaptation



conf = {
    "patch_shape": [800, 800],
    "difficulty": 0.8,
    "translation": 1.0,
    "n_angles": 10,
    "max_angle": 60,
    "min_convexity": 0.05,
}

sp_conf = {
    "max_num_keypoints": None,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}

homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}


def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': True,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'train_batch_size': 1,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'train'  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'train'))
    return loader



def process_image(img_data, num_H, output_folder_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img_data["image"].to(device)  # B x C x H x W
    net = DeepLSD({}).to(device)
        # Run homography adaptation
    distance_field, angle_field, _ = generate_ground_truth_with_homography_adaptation(img,net, num_H=num_H)

    assert len(img_data["name"]) == 1  # Currently expect batch size one!
    # store gt in same structure as images of minidepth
    
    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)
    output_file_path = complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}.hdf5"
    
    # Save the DF in a hdf5 file
    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("deeplsd_distance_field", data=distance_field)
        f.create_dataset("deeplsd_angle_field", data=angle_field)


def export_ha(data_loader, output_folder_path, num_H, n_jobs):
    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(process_image)(img_data, num_H, output_folder_path) for img_data in
        tqdm(data_loader, total=len(data_loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Output folder.', default="superpoint_gt")
    parser.add_argument('--num_H', type=int, default=100, help='Number of homographies used during HA.')
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of jobs (that perform HA) to run in parallel.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1, help='Number of jobs the dataloader uses to load images')
    args = parser.parse_args()

    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)

    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader)
    export_ha(dataloader, out_folder_path, args.num_H, args.n_jobs)
    print("Done !")
