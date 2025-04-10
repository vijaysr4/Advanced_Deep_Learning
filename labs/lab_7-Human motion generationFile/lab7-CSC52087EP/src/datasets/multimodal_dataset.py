from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

from utils.file_utils import load_txt

# ------------------------------------------------------------------------------------- #


class MultimodalDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        dataset_name: str,
        human: Dataset,
        caption: Dataset,
        num_frames: int,
        num_feats: int,
        num_cond_feats: int,
        standardization: Dict[str, Any] = None,
    ):
        self.name = name

        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name

        self.num_frames = num_frames
        self.num_feats = num_feats
        self.num_cond_feats = num_cond_feats

        self.human_dataset = human
        self.caption_dataset = caption
        self.standardization = standardization

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str):
        self.split = split
        split_path = Path(self.dataset_dir) / f"humanml3d_{split}_split.txt"
        sample_ids = load_txt(split_path).split("\n")
        self.sample_ids = sorted(sample_ids)

        self.human_dataset.split = self.split
        self.human_dataset.sample_ids = self.sample_ids
        self.get_body = self.human_dataset.get_rifkebody
        self.normalize = self.human_dataset.normalize
        self.unnormalize = self.human_dataset.unnormalize

        self.caption_dataset.split = self.split
        self.caption_dataset.sample_ids = self.sample_ids

        return self

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> Dict[str, Any]:
        out = {"index": index}

        human_out = self.human_dataset[index]
        sample_id, human_feature, padding_mask = human_out
        out["sample_id"] = sample_id
        out["human_feat"] = human_feature
        out["padding_mask"] = padding_mask

        caption_out = self.caption_dataset[index]
        sample_id, caption_feature, raw_caption = caption_out
        out["sample_id"] = sample_id
        out["caption_feat"] = caption_feature
        out["caption_raw"] = raw_caption

        return out

    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        index = self.sample_ids.index(sample_id)
        return self[index]

    def __len__(self):
        return len(self.human_dataset)
