import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.jpldd.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.extractors.jpldd.descriptor_head import SDDH
from gluefactory.models.extractors.jpldd.keypoint_decoder import SMH
from gluefactory.models.extractors.jpldd.keypoint_detection import DKD
from gluefactory.models.extractors.jpldd.utils import InputPadder, change_dict_key

to_ctr = OmegaConf.to_container  # convert DictConfig to dict
aliked_checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"
logger = logging.getLogger(__file__)


def renormalize_keypoints(keypoints, img_wh):
    if isinstance(keypoints, torch.Tensor):
        return img_wh * (keypoints + 1.0) / 2.0
    elif isinstance(keypoints, list):
        for i in range(len(keypoints)):
            keypoints[i] = img_wh * (keypoints[i] + 1.0) / 2.0
        return keypoints


class JointPointLineDetectorDescriptor(BaseModel):
    # currently contains only ALIKED
    default_conf = {
        # ToDo: create default conf once everything is running -> default conf is merged with input conf to the init method!
        "model_name": "aliked-n16",
        "max_num_keypoints": 1000,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 2,
        "timeit": True,  # override timeit: False from BaseModel
        "train_descriptors": {
            "do": True,  # if train is True, initialize ALIKED Light model form OTF Descriptor GT
            "device": None  # device to house the lightweight ALIKED model
        },
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect!

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"final config dict(type={type(conf)}): {conf}")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        aliked_model_cfg = aliked_cfgs[conf.model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        # Load Network Components
        self.encoder_backbone = AlikedEncoder(aliked_model_cfg)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        self.dkd = DKD(radius=conf.nms_radius,
                       top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
                       scores_th=conf.detection_threshold,
                       n_limit=(
                           conf.max_num_keypoints
                           if conf.max_num_keypoints > 0
                           else self.n_limit_max
                       ), )  # Differentiable Keypoint Detection from ALIKE
        # Keypoint and line descriptors
        self.descriptor_branch = SDDH(dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False)
        self.line_descriptor = torch.lerp  # we take the endpoints of lines and interpolate to get the descriptor
        # Line Attraction Field information (Line Distance Field and Angle Field)
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.angle_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # ToDo Figure out heuristics
        # self.line_extractor = LineExtractor(torch.device("cpu"), self.line_extractor_cfg)

        if conf.timeit:
            self.timings = {
                "total-makespan": [],
                "encoder": [],
                "keypoint-and-junction-heatmap": [],
                "line-af": [],
                "line-df": [],
                "descriptor-branch": [],
                "keypoint-detection": []
            }

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.pretrained:
            logger.info("Load pretrained weights for aliked parts...")
            old_test_val1 = self.encoder_backbone.conv1.weight.data.clone()
            self.load_pretrained_elements()
            assert not torch.all(torch.eq(self.encoder_backbone.conv1.weight.data.clone(),
                                          old_test_val1)).item()  # test if weights really loaded!

        # Initialize Lightweight ALIKED model to perform OTF GT generation for descriptors if training
        if conf.train_descriptors.do:
            logger.info("Load ALiked Lightweight model for descriptor training...")
            device = conf.train_descriptors.device if conf.train_descriptors.device is not None else (
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.aliked_lw = get_model("jpldd.aliked_light")(aliked_model_cfg).eval().to(
                device)  # use same config than for our network parts

    # Utility methods for line df and af with deepLSD
    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        """
        if self.conf.timeit:
            total_start = time.time()
        # output container definition
        output = {}

        # load image and padder
        image = data["image"]
        div_by = 2 ** 5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        padded_img = padder.pad(image)

        # pass through encoder
        if self.conf.timeit:
            start_encoder = time.time()
            feature_map_padded = self.encoder_backbone(padded_img)
            self.timings["encoder"].append(time.time() - start_encoder)
        else:
            feature_map_padded = self.encoder_backbone(padded_img)

        # pass through keypoint & junction decoder
        if self.conf.timeit:
            start_keypoints = time.time()
            score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
            self.timings["keypoint-and-junction-heatmap"].append(time.time() - start_keypoints)
        else:
            score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)

        # normalize and remove padding and format dimensions
        feature_map_padded_normalized = torch.nn.functional.normalize(feature_map_padded, p=2, dim=1)
        feature_map = padder.unpad(feature_map_padded_normalized)
        logger.debug(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}")
        assert (feature_map.shape[2], feature_map.shape[3]) == (image.shape[2], image.shape[3])
        keypoint_and_junction_score_map = padder.unpad(score_map_padded)  # B x 1 x H x W
        output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map.squeeze()  # B x H x W

        # Line AF Decoder
        if self.conf.timeit:
            start_line_af = time.time()
            line_angle_field = self.angle_field_branch(feature_map)
            self.timings["line-af"].append(time.time() - start_line_af)
        else:
            line_angle_field = self.angle_field_branch(feature_map)

        # Line DF Decoder
        if self.conf.timeit:
            start_line_df = time.time()
            line_distance_field = self.distance_field_branch(feature_map)
            self.timings["line-df"].append(time.time() - start_line_df)
        else:
            line_distance_field = self.distance_field_branch(feature_map)

        output[
            "deeplsd_line_anglefield"] = line_angle_field.squeeze()  # squeeze to remove size 1 dim to match groundtruth
        output["deeplsd_line_distancefield"] = line_distance_field.squeeze()

        # Keypoint detection
        if self.conf.timeit:
            start_keypoints = time.time()
            keypoints, kptscores, scoredispersitys = self.dkd(
                keypoint_and_junction_score_map,
            )
            self.timings["keypoint-detection"].append(time.time() - start_keypoints)
        else:
            keypoints, kptscores, scoredispersitys = self.dkd(
                keypoint_and_junction_score_map,
            )
        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required,
        # we can set detection_threshold=-1 and conf.max_num_keypoints -> HERE WE SET THESE VALUES SO WE CAN EXPECT SAME NUM!
        output["keypoints"] = wh * (torch.stack(
            keypoints) + 1) / 2.0  # renormalize_keypoints(keypoints, wh)  # B N 2 (list of B tensors having N by 2)
        output["keypoint_scores"] = torch.stack(kptscores),  # B N
        output["keypoint_score_dispersity"] = torch.stack(scoredispersitys),

        # Keypoint descriptors
        if self.conf.timeit:
            start_desc = time.time()
            keypoint_descriptors, offsets = self.descriptor_branch(feature_map, keypoints)
            self.timings["descriptor-branch"].append(time.time() - start_desc)
        else:
            keypoint_descriptors, offsets = self.descriptor_branch(feature_map, keypoints)
        output["keypoint_descriptors"] = torch.stack(keypoint_descriptors)  # B N D

        # Extract Lines from Learned Part of the Network
        # Only Perform line detection when NOT in training mode
        if not self.training:
            line_segments = None  # as endpoints
            output["line_segments"] = line_segments
            # Use aliked points sampled from inbetween Line endpoints?
            line_descriptors = None
            output["line_descriptors"] = line_descriptors

        if self.conf.timeit:
            self.timings["total-makespan"].append(time.time() - total_start)
        return output

    def loss(self, pred, data):
        """
        perform loss calculation based on prediction and data(=groundtruth) for a batch
        1. On Keypoint-ScoreMap:        L1/L2 Loss / FC-Softmax?
        2. On Keypoint-Descriptors:     L1/L2 loss
        3. On Line-Angle Field:         L1/L2 Loss / FC-Softmax?
        4. On Line-Distance Field:      L1/L2 Loss / FC-Softmax?
        """
        losses = {}
        metrics = {}

        # calculate losses and store them into dict
        keypoint_scoremap_loss = F.l1_loss(pred["keypoint_and_junction_score_map"],
                                           data["superpoint_heatmap"], reduction='mean')
        losses["keypoint_and_junction_score_map"] = keypoint_scoremap_loss
        # Descriptor Loss: expect aliked descriptors as GT
        if self.conf.train_descriptors.do: # todo: compute descr gt here!
            keypoint_descriptor_loss = F.l1_loss(pred["keypoint_descriptors"], data["aliked_descriptors"], reduction='mean')
            losses["keypoint_descriptors"] = keypoint_descriptor_loss
        line_af_loss = F.l1_loss(pred["deeplsd_line_anglefield"], data["deeplsd_angle_field"], reduction='mean')
        losses["deeplsd_line_anglefield"] = line_af_loss
        line_df_loss = F.l1_loss(pred["deeplsd_line_distancefield"], data["deeplsd_distance_field"], reduction='mean')
        losses["deeplsd_line_distancefield"] = line_df_loss

        # Todo: different weightings
        overall_loss = keypoint_scoremap_loss + line_af_loss + line_df_loss
        if self.conf.train_descriptors.do:
            overall_loss += keypoint_descriptor_loss
        losses["total"] = overall_loss

        # add metrics if not in training mode
        if not self.training:
            metrics = self.metrics(pred, data)
        return losses, metrics

    def get_groundtruth_descriptors(self, pred: dict):
        """
        Takes keypoints from predictions (best 100 + 100 random) + computes ground-truth descriptors for it.
        """
        assert pred.get('image', None) is not None and pred.get('keypoints', None) is not None  # todo: check dims
        with torch.no_grad():
            descriptors = self.aliked_lw(pred)
        return descriptors

    def load_pretrained_elements(self):
        """
        Loads ALIKED weights for backbone encoder, score_head(SMH) and SDDH
        """
        # Load state-dict of wanted aliked-model
        aliked_state_url = aliked_checkpoint_url.format(self.conf.model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(aliked_state_url, map_location="cpu")
        # change keys
        for k, v in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                change_dict_key(aliked_state_dict, k, f"keypoint_and_junction_branch.{k}")
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue
        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_loss_keys_in_dict(self, data_keys):
        for required_loss_key in self.conf.required_loss_keys:
            if required_loss_key not in data_keys:
                return False
        return True

    def state_dict(self, *args, **kwargs):
        """
        Custom state dict to exclude aliked_lw module from checkpoint.
        """
        sd = super().state_dict(*args, **kwargs)
        del sd["aliked_lw"]
        return sd

    def get_current_timings(self, reset=False):
        """
        ONLY USE IF TIMEIT ACTIVATED. It returns the average of the current times in a dictionary for
        all the single network parts.

        reset: if True deletes all collected times until now
        """
        results = {}
        for k, v in self.timings.items():
            results[k] = np.mean(v)
            if reset:
                self.timings[k] = []
        return results

    def get_pr(self, pred_kp: torch.Tensor, gt_kp: torch.Tensor, tol=3): # todo, make it work!
        """ Compute the precision and recall, based on GT KP. """
        if len(gt_kp) == 0:
            precision = float(len(pred_kp) == 0)
            recall = 1.
        elif len(pred_kp) == 0:
            precision = 1.
            recall = float(len(gt_kp) == 0)
        else:
            dist = torch.norm(pred_kp[:, None] - gt_kp[None], dim=2)
            close = (dist < tol).float()
            precision = close.max(dim=1)[0].mean()
            recall = close.max(dim=0)[0].mean()
        return precision, recall

    def metrics(self, pred, data):
        device = pred['keypoint_and_junction_score_map'].device
        gt_keypoints = data["superpoint_heatmap"] > 0

        # Compute the precision and recall
        precision, recall = [], []
        for i in range(len(data['superpoint_heatmap'])):  # iter over batch dim
            valid_gt_kp = data['superpoint_heatmap'][i][gt_keypoints[i]]
            #precision, recall = self.get_pr(pred['keypoints'][i], valid_gt_kp)
            precision, recall = 0.5, 0.5
            precision.append(precision)
            recall.append(recall)

        # Compute the KP repeatability and localization error
        #rep, loc_error = get_repeatability_and_loc_error(
        #    pred['keypoints0'], pred['keypoints1'], pred['keypoint_scores0'],
        #    pred['keypoint_scores1'], data['H_0to1'])

        out = {
            'precision': torch.tensor(precision, dtype=torch.float, device=device),
            'recall': torch.tensor(recall, dtype=torch.float, device=device),
         #   'repeatability': rep, 'loc_error': loc_error
        }
        return out
