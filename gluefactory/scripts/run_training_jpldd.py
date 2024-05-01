"""
File to test and research how JPLDD model behaves during training.

- Works with one GPU currently

Tests:
- checkpoint load & store
- train with descriptors in parallel
- checkout config
"""

import argparse
import logging

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models import get_model

logger = logging.getLogger(__name__)

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
    "train_descriptors": True
}
train_conf = OmegaConf.create(default_train_conf)


def get_dataset_and_loader(num_dataloader_workers):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': False,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'batch_size': 4,
        'train_batch_size': 2,  # prefix must match split mode
        'num_workers': num_dataloader_workers,
        'split': 'train',  # if implemented by dataset class gives different splits
        "load_features": {
            "do": True,
            "device": None,  # choose device to move groundtruthdata to if None is given, just read, skip move to device
            "point_gt": {
                "path": "outputs/results/superpoint_gt",
                "data_keys": ["superpoint_heatmap"]
            },
            "line_gt": {
                "path": "outputs/results/deeplsd_gt",
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"]
            }
        },
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'train'), pinned=True, distributed=False)
    return dataset, loader


def get_model_object(model_name, device):
    model_specific_config = {  # define model specific config
        'pretrained': True
    }
    omega_ms_conf = OmegaConf.create(model_specific_config)
    model = get_model(model_name)(omega_ms_conf).to(device)
    return model


def run(model_name, num_workers_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Load dataset and create dataloader...")
    dataset, data_loader = get_dataset_and_loader(num_dataloader_workers=num_workers_dataloader)
    print(f"Load model: {model_name}")
    model = get_model_object(model_name, device)
    train(model, data_loader)


def train(model, dataloader):
    model = model.train()
    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[train_conf.optimizer]
    optimizer = optimizer_fn(model.parameters())
    for i in tqdm(range(train_conf.epochs), desc="Epochs", unit="epoch"):
        for batch in tqdm(dataloader, desc="Batches", unit="batch"):
            optimizer.zero_grad()
            pred = model(batch)
            img_and_keypoints = {"keypoints": pred["keypoints"], "image": batch["image"]}
            batch = {**batch, **model.get_groundtruth_descriptors(img_and_keypoints)}
            losses, _ = model.loss(pred, batch)
            loss = losses["total"].mean()
            if torch.isnan(loss).any():
                print(f"Detected NAN, skipping iteration")
                del pred, batch, loss, losses
                continue
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)  # model to load and export features with
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    logger.info("Running Export...")
    run(args.model, args.num_workers)
    logger.info("Done!")

"""
TODO: 1. finish training loop -> epochs, basic prints
      2. add descriptor groundtruth generation to loop OR model (probably model if want to use train.py)
      3. Test loss functions and update
      3.5 incorperate validation
      4. Add timings to network & test tensorboard
"""
