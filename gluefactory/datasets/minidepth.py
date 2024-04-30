import shutil
import zipfile
from pathlib import Path

import h5py
import torch
import logging


from gluefactory.datasets import BaseDataset
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import DATA_PATH
from gluefactory.utils.image import load_image, ImagePreprocessor

logger = logging.getLogger(__name__)


class MiniDepthDataset(BaseDataset):
    """
    Assumes minidepth dataset in folder as jpg images
    """
    default_conf = {
        "data_dir": "minidepth/images",  # as subdirectory of DATA_PATH(defined in settings.py)
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "device": None,  # specify device to move image data to. if None is given, just read, skip move to device
        "split": "train",
        "seed": 0,
        "preprocessing": {
            'resize': [800, 800]
        },
        "load_features": {
            "do": False,
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

    def _init(self, conf):
        self.grayscale = bool(conf.grayscale)
        # self.conf is set in superclass

        # set img preprocessor
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # Auto-download the dataset
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the minidepth dataset...")
            self.download_minidepth()

        # Form pairs of images from the multiview dataset
        self.img_dir = DATA_PATH / conf.data_dir
        # load all image paths
        self.image_paths = list(Path(self.img_dir).glob("**/*.jpg"))
        # making them relative for system independent names in export files (path used as name in export)
        self.image_paths = [i.relative_to(self.img_dir) for i in self.image_paths.copy()]
        if len(self.image_paths) == 0:
            raise ValueError(
                f"Could not find any image in folder: {self.img_dir}."
            )
        logger.info(f"NUMBER OF IMAGES: {len(self.image_paths)}")
        # Load features
        if conf.load_features.do:
            self.point_gt_location = DATA_PATH / conf.load_features.point_gt.path
            self.line_gt_location = DATA_PATH / conf.load_features.line_gt.path

    def download_minidepth(self):
        logger.info("Downloading the MiniDepth dataset...")
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "minidepth_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://filedn.com/lt6zb4ORSwapNyVniJf1Pqh/"
        zip_name = "minidepth.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url_base + zip_name, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / zip_name.split(".")[0], data_dir)

    def get_dataset(self, split):
        return self

    def _read_image(self, path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img = load_image(path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_groundtruth(self, image_path, enforce_batch_dim=True):
        """
        Reads groundtruth for points and lines from respective h5files

        image_path: path to image as relative to base directory(self.img_path)
        """
        ground_truth = {}
        point_gt_file_path = self.point_gt_location / image_path
        line_gt_file_path = self.line_gt_location / image_path
        assert point_gt_file_path.exists() and line_gt_file_path.exists()
        # Read data for points
        with h5py.File(point_gt_file_path, "r") as point_file:
            ground_truth = {**self.read_datasets_from_h5(self.conf.load_features.point_gt.data_keys, point_file),
                            **ground_truth}
        # Read data for lines
        with h5py.File(line_gt_file_path, "r") as line_file:
            ground_truth = {**self.read_datasets_from_h5(self.conf.load_features.line_gt.data_keys, line_file),
                            **ground_truth}
        # todo: to tensor / batch handling (is this handled by dataset or loader??)
        return ground_truth

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        path = self.image_paths[idx]
        img = self._read_image(self.img_dir / path)
        data = {"name": str(path), **self.preprocessor(img)}  # add metadata, like transform, image_size etc...
        if self.conf.load_features.do:
            gt = self._read_groundtruth(path)
            data = {**data, **gt}
        # fix err in dkd todo check together with batching
        # del data['image_size']  # torch.from_numpy(data['image_size'])
        return data

    def read_datasets_from_h5(self, keys, file):
        data = {}
        for key in keys:
            d = file[key]
            if self.conf.load_features.device is not None:
                data[key] = d.to(self.conf.load_features.device)
        return data

    def __len__(self):
        return len(self.image_paths)
