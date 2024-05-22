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
import time

from omegaconf import OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed

from gluefactory.settings import EVAL_PATH
from gluefactory.datasets import get_dataset
from gluefactory.models.extractors.superpoint_open import SuperPoint
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.utils.image import numpy_image_to_torch

conf = {
    "max_num_keypoints": None,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}

conf_aliked = {
    "name": 'extractors.aliked',
    "trainable": False,
    "max_num_keypoints": -1,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "trainings": {'do' : False},
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}


conf_lines = {
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
}

sptime = []
alikedtime = []
jplddtime = []
deeplsdtime = []


def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': False,  # commented out things -> dataset must also have these keys but has not
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_s', type=int, default=100, help='Number of timing samples.')
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of jobs (that perform HA) to run in parallel.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1, help='Number of jobs the dataloader uses to load images')
    args = parser.parse_args()


    print("NUMBER OF SAMPLES: ", args.num_s)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)
    
    to_ctr = OmegaConf.to_container 

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modelsp = SuperPoint(conf).to(device)
    modelsp.eval().to(device)


    conf_aliked_omega = OmegaConf.create(conf_aliked)

    modelaliked = get_model('extractors.aliked')
    modelaliked = modelaliked(to_ctr(conf_aliked_omega)).to(device)

    modeljpldd = get_model('extractors.jpldd.joint_point_line_extractor')
    modeljpldd = modeljpldd(to_ctr(conf_aliked_omega)).to(device)


    modelDeepLSD = DeepLSD(conf_lines)
    modelDeepLSD.eval().to(device)
    count = 0
    
    for img in tqdm(dataloader):
        #image_data = {'image': numpy_image_to_torch(img)[None].to(device)}
        image_data = img["image"].to(device)
        
        # Check the shape of the image tensor
        if image_data.shape[1] == 1:
            # Expand the single channel to three channels
            image_data = image_data.repeat(1, 3, 1, 1)
        
        # Convert the image tensor to a NumPy array
        img_npy = image_data.cpu().detach().numpy()  # Ensure the tensor is on the CPU and detached before converting to NumPy
        img_npy = img_npy[0, :, :, :]  # Remove the batch dimension
        img_npy = np.transpose(img_npy, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Convert to uint8 and rescale pixel values if necessary
        image_data_uint8 = (img_npy * 255).astype(np.uint8)
        
        # Convert the image to grayscale
        img_npy_gray = cv2.cvtColor(image_data_uint8, cv2.COLOR_BGR2GRAY)
        
        # Continue with your processing, re-creating the image dictionary for the model
        img_gray = {"image": numpy_image_to_torch(img_npy_gray)[None].to(device)}
        img = {"image": numpy_image_to_torch(img_npy)[None].to(device)}
        
        torch.cuda.synchronize()
        
        start = time.time()    
        pred = modelsp(img_gray)
        end = time.time()
        sptime.append((end - start))
        
        start = time.time()
        pred = modelaliked(img)
        end = time.time()
        alikedtime.append((end - start))
        
        
        start = time.time()
        pred = modeljpldd(img)
        end = time.time()
        jplddtime.append((end - start))
        
        start = time.time()
        pred = modelDeepLSD(img)
        end = time.time()
    
        deeplsdtime.append((end - start))
        
        count += 1
        
        if count == args.num_s:
            break
        
    
    print("SuperPoint: ", np.mean(sptime))
    print("ALIKED: ", np.mean(alikedtime))
    print("JPLDD: ", np.mean(jplddtime))
    print("DeepLSD: ", np.mean(deeplsdtime))
    
    print("Done !")
