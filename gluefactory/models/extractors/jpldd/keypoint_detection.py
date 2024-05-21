"""
keypoint_detection.py: Contains nets / methods that extract keypoints from the keypoint score map
- Simple NMS + Thresholding
"""

import torch
from .utils import simple_nms


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class SimpleDetector(object):
    def __init__(self, nms_radius: int, num_keypoints: int, threshold: float):
        self.nms_radius = nms_radius
        self.num_keypoints = num_keypoints
        self.threshold = threshold

    def __call__(self, keypoint_heatmap: torch.Tensor):
        b, c, h, w = keypoint_heatmap.shape
        nms_scores = simple_nms(keypoint_heatmap, self.nms_radius)  # B x c x h x w
        # remove border
        nms_scores[:, :, : self.nms_radius, :] = 0
        nms_scores[:, :, :, : self.nms_radius] = 0
        nms_scores[:, :, -self.nms_radius:, :] = 0
        nms_scores[:, :, :, -self.nms_radius:] = 0

        # select top keypoints or threshold
        # Extract keypoints
        if b > 1:
            idxs = torch.where(nms_scores > self.threshold)
            mask = idxs[0] == torch.arange(b, device=nms_scores.device)[:, None]
        else:  # Faster shortcut
            nms_scores = nms_scores.squeeze(0)
            idxs = torch.where(nms_scores > self.threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = nms_scores[idxs]

        keypoints = []
        scores = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.num_keypoints)

            keypoints.append(k)
            scores.append(s)
        return keypoints, scores
