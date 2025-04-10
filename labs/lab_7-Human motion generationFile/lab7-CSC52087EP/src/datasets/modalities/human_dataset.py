from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from smplx import build_layer
from smplx.utils import SMPLXOutput
import torch
from torchtyping import TensorType
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.rifke_utils import smplrifkefeats_to_smpldata, smpldata_to_bodymodel

# ------------------------------------------------------------------------------------- #

num_batches, num_frames, num_raws, num_feats = None, None, None, None

# ------------------------------------------------------------------------------------- #


class HumanDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_feats: int,
        num_frames: int,
        standardization: Dict[str, Any],
        sequential: bool,
        **kwargs,
    ):
        super().__init__()

        self.modality = "human"
        self.name = name

        self.dataset_dir = Path(dataset_dir)
        self.feat_dir = self.dataset_dir / "smpl_rifke"

        self.num_feats = num_feats
        self.num_frames = num_frames
        self.sequential = sequential

        self.feat_mean = torch.load(standardization["feat_rifke"]["mean_path"])
        self.feat_std = torch.load(standardization["feat_rifke"]["std_path"])
        self.tmrrifke_mean = torch.load(standardization["tmr_rifke"]["mean_path"])
        self.tmrrifke_std = torch.load(standardization["tmr_rifke"]["std_path"])

        self.body_model = build_layer(**kwargs["smpl_kwargs"])
        self.body_model.eval()
        for p in self.body_model.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.sample_ids)

    # --------------------------------------------------------------------------------- #

    @torch.no_grad()
    def get_rifkebody(
        self, human_feature: TensorType["num_frames", "num_feats"]
    ) -> SMPLXOutput:
        human_raw = human_feature.clone()
        self.body_model.to(human_raw.device)

        human_raw *= self.feat_std.to(human_raw.device)
        human_raw += self.feat_mean.to(human_raw.device)

        smpl_joints, smpl_poses = smplrifkefeats_to_smpldata(human_raw)
        bodymodel_out = smpldata_to_bodymodel(self.body_model, smpl_joints, smpl_poses)
        bodymodel_out.faces = torch.from_numpy(self.body_model.faces).to(
            human_raw.device
        )

        return bodymodel_out

    # --------------------------------------------------------------------------------- #

    def normalize(
        self, raw_feat: TensorType["num_frames", "num_raws"], feat_name: str
    ) -> TensorType["num_frames", "num_feats"]:
        norm_feat = raw_feat.clone()

        norm_feat -= getattr(self, f"{feat_name}_mean").to(raw_feat.device)
        norm_feat /= getattr(self, f"{feat_name}_std").to(raw_feat.device)

        return norm_feat

    def unnormalize(
        self, norm_feat: TensorType["num_frames", "num_feats"], feat_name: str
    ) -> TensorType["num_frames", "num_feats"]:
        raw_feat = norm_feat.clone()

        raw_feat *= getattr(self, f"{feat_name}_std").to(norm_feat.device)
        raw_feat += getattr(self, f"{feat_name}_mean").to(norm_feat.device)

        return raw_feat

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> Tuple[
        str,
        TensorType["num_feats", "num_frames"],
        TensorType["num_frames", "num_raws"],
        TensorType["num_frames"],
    ]:
        sample_id = self.sample_ids[index]

        feat_filename = sample_id + ".npy"
        feat_path = self.feat_dir / feat_filename

        human_raw = torch.from_numpy(np.load((feat_path)))
        human_raw = human_raw.to(torch.float32)[: self.num_frames]
        human_feature = self.normalize(human_raw, "feat")
        padding_size = self.num_frames - human_raw.shape[0]
        padded_human_feature = F.pad(human_feature, (0, 0, 0, padding_size))

        # if not self.sequential:
        #     padded_human_feature = padded_human_feature.reshape(-1)

        # 1 for valid cams, 0 for padded cams
        padding_mask = torch.ones((self.num_frames))
        padding_mask[human_feature.shape[0] :] = 0

        return sample_id, padded_human_feature, padding_mask
