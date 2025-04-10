from typing import Dict, List

import torch
from torchmetrics import Metric
from torchtyping import TensorType

# -------------------------------------------------------------------------------------- #

num_samples, num_feats, num_topk = None, None, None

# -------------------------------------------------------------------------------------- #


def calculate_top_k(
    pred_indices: TensorType["num_samples", "num_samples"], top_k: int
) -> TensorType["num_samples", "num_topk"]:
    num_samples = pred_indices.shape[0]
    target_indices = torch.arange(num_samples, device=pred_indices.device)
    target_indices = target_indices[:, None].expand(-1, num_samples)
    bool_indices = pred_indices == target_indices
    bool_top_k = torch.cumsum(bool_indices[:, :top_k], dim=1).bool()
    return bool_top_k


class MotionTextMetric(Metric):
    def __init__(self, top_k: List[int], **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

        self.add_state(
            "top_k_count",
            default=torch.zeros(max(self.top_k)).long(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_num_samples", default=torch.tensor(0).long(), dist_reduce_fx="sum"
        )
        self.add_state(
            "mm_distance_sum",
            default=torch.tensor(0).double(),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        char_feat: TensorType["num_samples", "num_feats"],
        text_feats: TensorType["num_samples", "num_feats"],
    ):
        """Update state with new character and text features."""
        # text_feats = torch.nn.functional.normalize(text_feats, p=2, dim=-1)
        # char_feat = torch.nn.functional.normalize(char_feat, p=2, dim=-1)

        char_text_cdist = torch.cdist(text_feats, char_feat)
        char_text_argsort = torch.argsort(char_text_cdist, dim=1)
        top_k_mat = calculate_top_k(char_text_argsort, top_k=max(self.top_k))

        self.top_k_count += top_k_mat.sum(0)
        self.total_num_samples += text_feats.shape[0]
        self.mm_distance_sum += char_text_cdist.trace()

    def compute(self) -> Dict[str, float]:
        """Compute retrieval precisons and multimodal distance."""
        retrieval_precision = self.top_k_count / self.total_num_samples
        mm_distance = self.mm_distance_sum / self.total_num_samples

        metric_dict = {f"R{k}": retrieval_precision[k - 1] for k in self.top_k}
        metric_dict["mm_distance"] = mm_distance

        return metric_dict
