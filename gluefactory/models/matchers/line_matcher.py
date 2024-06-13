from ...geometry.gt_generation import (
    gt_line_matches_from_homography,
    gt_matches_from_homography,
)
from ..base_model import BaseModel
from gluefactory.models.extractors.jpldd.metrics_lines import match_segments_1_to_1
import numpy as np


class HomographyMatcher(BaseModel):
    default_conf = {
        "img_size": (800,800),
        "line_dist": "area",
        "angular_th": (30 * np.pi / 180),
        "overlap_th": 0.5,
        "dist_thresh": 0.5
    }

    required_data_keys = ["H_0to1","lines0","lines1"]

    def _forward(self, data):
        result = {}
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(data["lines0"],data["lines1"],data["H_0to1"],
                              self.conf.img_size,self.conf.line_dist,self.conf.angular_th,
                              self.conf.overlap_th,self.conf.dist_thresh)
        result["lines0"] = segs1
        result["lines1"] = segs2
        result["line_matches0"] = matched_idx1
        result["line_matches1"] = matched_idx2
        result["line_distances"] = distances
        return result

    def loss(self, pred, data):
        raise NotImplementedError
