"""
Run the homography adaptation with DeepLSD for all images in the minidepth dataset.
Goal: create groundtruth with DeepLSD. Format: stores groundtruth for every image in a separate file.
"""

import argparse
from pathlib import Path
from ctypes import c_char_p

import h5py
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler, DataLoader
import os
from multiprocessing import Value, Manager, Array

from gluefactory.settings import EVAL_PATH, DATA_PATH
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


def get_dataset_and_loader(num_workers, rank_distributed: int = None, world_size: int = None):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': True,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'test_batch_size': 1,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'test',  # if implemented by dataset class gives different splits
        "prefetch_factor": None if num_workers == 0 else 2,
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    sampler = DistributedSampler(dataset, rank=rank_distributed, num_replicas=world_size) if rank_distributed else None
    loader = DataLoader(dataset, batch_size=config['test_batch_size'], sampler=sampler, num_workers=num_workers, shuffle=False)
    return loader


def process_image(img_data, net, num_H, output_folder_path, device):
    img = img_data["image"].to(device)  # B x C x H x W
    # Run homography adaptation
    distance_field, angle_field, _ = generate_ground_truth_with_homography_adaptation(img, net, num_H=num_H, bs=8)
    assert len(img_data["name"]) == 1, f"Image data name is {img_data['name']}"  # Currently expect batch size one!
    # store gt in same structure as images of minidepth
    img_name = img_data["name"][0]
    complete_out_folder = (output_folder_path / img_name).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)
    output_file_path = complete_out_folder / f"{Path(img_name).name.split('.')[0]}.hdf5"

    # Save the DF in a hdf5 file
    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("deeplsd_distance_field", data=distance_field)
        f.create_dataset("deeplsd_angle_field", data=angle_field)


def export_ha(output_folder_path, num_H, n_gpus, image_name_list_filepath):
    if n_gpus > 1:
        lock = mp.Lock()
        shared_out_folder_path = Array("c", str(output_folder_path).encode())
        shared_numH = Value("i", int(num_H))
        shared_img_name_list = Array("c", str(image_name_list_filepath).encode())
        shared_nGPUs = Value("i", n_gpus)
        # mp.spawn will induce the rank as argument
        mp.spawn(export_ha_parallel, args=(shared_nGPUs, shared_out_folder_path, shared_numH, shared_img_name_list, lock), nprocs=n_gpus,
                 join=True)
    else:
        raise NotImplementedError


def export_ha_parallel(rank, world_size, output_folder_path, num_H, image_name_list_filepath, lock):
    dist.init_process_group("nccl", rank=rank, world_size=world_size.value)
    data_loader = get_dataset_and_loader(0, rank, world_size.value)  # creates distributed dataloader for each process
    device = f"cuda:{rank}"
    with open(image_name_list_filepath.value.decode(), "r") as f:
        image_list = f.readlines()
    image_list = [elem[:-1] for elem in image_list]
    net = DeepLSD({}).to(device)
    for img_data in data_loader:
        if img_data["name"][0] in image_list:
            continue
        process_image(img_data, net, num_H.value, output_folder_path.value.decode(), device)
        lock.acquire()
        with open(image_name_list_filepath.value.decode(), "a") as f:
            f.write(img_data["name"][0] + "\n")
        lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Output folder.', default="deeplsd_gt")
    parser.add_argument('--num_H', type=int, default=100, help='Number of homographies used during HA.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1,
                        help='Number of jobs the dataloader uses to load images')
    parser.add_argument("--n_gpus", type=int, default=0, help="How many gpus we can use")
    parser.add_argument("--image_name_list", type=str,
                        help="File with list of names of images that have been generated, relative to our team folder")
    args = parser.parse_args()
    # image name list read/create
    image_name_list_filepath = DATA_PATH / args.image_name_list
    if not os.path.exists(image_name_list_filepath):
        with open(image_name_list_filepath, "w"): pass

    # create outputfolder if not exisitng
    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)

    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)
    print("N GPUS: ", args.n_gpus)

    export_ha(out_folder_path, args.num_H, args.n_gpus, image_name_list_filepath)
    print("Done !")
