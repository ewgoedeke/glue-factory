import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.jpldd.metrics_points import compute_pr, compute_loc_error, compute_repeatability
import gluefactory.models.extractors.jpldd.metrics_lines as LineMetrics
from gluefactory.datasets.homographies_deeplsd import sample_homography
from kornia.geometry.transform import warp_perspective
from gluefactory.models.extractors.jpldd.metrics_points import compute_pr, compute_loc_error, compute_repeatability
from gluefactory.models.extractors.jpldd.line_detection_lsd import detect_afm_lines
from gluefactory.models.extractors.jpldd.joint_point_line_extractor import JointPointLineDetectorDescriptor


logger = logging.getLogger(__file__)


class JPLDDEval(BaseModel):

    def _init(self, conf):
        self.model = JointPointLineDetectorDescriptor(conf)

    def _forward(self, data):
        normal_outputs = self.model({"image": data["image"]})
        warped_outputs,Hs = self.model._get_warped_outputs(data)
        return {
            "lines0": normal_outputs["lines"],
            "lines1": warped_outputs["lines"],
            "H": Hs
        }
