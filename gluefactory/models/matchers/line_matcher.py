from ...geometry.gt_generation import (
    gt_line_matches_from_homography,
    gt_matches_from_homography,
)
from ..base_model import BaseModel
from gluefactory.models.extractors.jpldd.metrics_lines import match_segments_1_to_1
import numpy as np
import torch


class LineMatcher(BaseModel):
    default_conf = {
        "img_size": (800,800),
        "line_dist": "area",
        "angular_th": (30 * np.pi / 180),
        "overlap_th": 0.5,
        "dist_thresh": 0.5
    }

    def _init(self, conf):
        # TODO (iago): Is this just boilerplate code?
        pass

    required_data_keys = ["H_0to1","lines0","lines1"]

    def _forward(self, data):
        device = data["lines0"][0].device
        result = {}
        # The data elements come in lists and therefore they are unpacked
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(data["lines0"][0].cpu().numpy(),
                                                                                    data["lines1"][0].cpu().numpy(),
                                                                                    data["H_0to1"][0].cpu().numpy(),
                              self.conf.img_size,self.conf.line_dist,self.conf.angular_th,
                              self.conf.overlap_th,self.conf.dist_thresh)
        result["lines0"] = torch.Tensor(segs1).to(device)
        result["lines1"] = torch.Tensor(segs2).to(device)
        result["line_matches0"] = torch.Tensor(matched_idx1).to(device)
        result["line_matches1"] = torch.Tensor(matched_idx2).to(device)
        result["line_matching_scores0"] = torch.Tensor(distances).to(device)
        result["line_matching_scores1"] = torch.Tensor(distances).to(device)
        return result

    def loss(self, pred, data):
        raise NotImplementedError
