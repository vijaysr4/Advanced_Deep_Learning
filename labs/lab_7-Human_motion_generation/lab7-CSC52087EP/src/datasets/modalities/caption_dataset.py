from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torchtyping import TensorType
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.file_utils import load_txt

# ------------------------------------------------------------------------------------- #

num_feats = None
MAX_CLIP_LENGTH = 77
MAX_DISTILBERT_LENGTH = 168

# ------------------------------------------------------------------------------------- #


class CaptionDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_frames: int,
        num_feats: int,
        sequential: bool,
        **kwargs,
    ):
        super().__init__()

        self.modality = "caption"
        self.name = name

        self.dataset_dir = Path(dataset_dir)

        self.caption_raw_dir = self.dataset_dir / "caption_raw"
        feat_dir = "seq" if sequential else "token"
        self.caption_feat_dir = self.dataset_dir / "caption_clip" / feat_dir
        self.clip_seq_dir = self.dataset_dir / "caption_clip" / "seq"

        self.num_frames = num_frames
        self.num_feats = num_feats
        self.sequential = sequential

    def __len__(self):
        return len(self.sample_ids)

    # --------------------------------------------------------------------------------- #

    def __getitem__(
        self, index: int
    ) -> Tuple[str, TensorType["num_feats"], Dict[str, torch.Tensor]]:
        filename = self.sample_ids[index]

        # Load data
        caption_feat_filename = self.caption_feat_dir / (filename + ".npy")
        caption_feat = np.load(caption_feat_filename, allow_pickle=True)
        # Pick random caption if multiple captions
        if self.split == "train":
            index = np.random.randint(0, len(caption_feat))
        else:
            index = 0
        caption_feat = caption_feat[index]
        caption_feat = torch.from_numpy(caption_feat.astype(np.float32))
        # Pad caption feature
        if self.sequential:
            feat_padding_size = MAX_CLIP_LENGTH - caption_feat.shape[0]
            caption_feat = F.pad(caption_feat, (0, 0, 0, feat_padding_size))

        # Load raw caption
        caption_raw_filename = self.caption_raw_dir / (filename + ".txt")
        caption_raw = load_txt(caption_raw_filename).splitlines()[index]

        # Load and prepare distilbert embeddings for TMR
        clip_filename = self.clip_seq_dir / (filename + ".npy")
        clip_caption = np.load(clip_filename, allow_pickle=True)[index]
        clip_caption = torch.from_numpy(clip_caption.astype(np.float32))
        padding_mask = torch.ones((MAX_CLIP_LENGTH))
        padding_mask[clip_caption.shape[0] :] = 0
        clip_padding_size = MAX_CLIP_LENGTH - clip_caption.shape[0]
        clip_caption = F.pad(clip_caption, (0, 0, 0, clip_padding_size))

        # Prepare data
        data_raw = {"caption": caption_raw}
        data_feat = caption_feat if caption_feat.dim() == 2 else caption_feat
        data_raw["clip_seq_caption"] = clip_caption
        data_raw["clip_seq_mask"] = padding_mask

        return filename, data_feat, data_raw
