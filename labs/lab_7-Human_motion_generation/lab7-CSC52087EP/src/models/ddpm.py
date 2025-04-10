from typing import List

import torch.nn as nn
from torchtyping import TensorType

# ----------------------------------------------------------------------------

num_batches, num_frames, num_feats, num_condfeats = None, None, None, None
Feats = TensorType["num_batches", "num_feats", "num_frames"]
Mask = TensorType["num_batches", "num_frames"]
Conds = TensorType["num_batches", "num_condfeats"]

# ----------------------------------------------------------------------------


class DDPMPrecond(nn.Module):
    def __init__(self, module: nn.Module, merger: nn.Module = None, **kwargs):
        super().__init__()

        self.model = module
        self.merger = merger
        self.num_frames = module.num_frames
        self.num_feats = module.num_feats

    def forward(
        self,
        gamma: TensorType["num_batches"],
        x: Feats,
        y: Conds = None,
        mask: Mask = None,
    ) -> List[Feats]:
        # Forward pass
        D_x = self.model(x, gamma.flatten(), y=y, mask=mask)
        return [D_x]
