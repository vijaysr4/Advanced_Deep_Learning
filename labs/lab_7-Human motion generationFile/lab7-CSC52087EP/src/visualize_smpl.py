from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from utils.rendering_utils import smpl_to_mp4

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    dataset = instantiate(config.dataset).set_split("test")
    sample = dataset[8]

    human_feat = sample["human_feat"]
    # --------------------------------------------------------------------------------- #
    # Complete this part for `Code 2`
    num_frames, num_feats = human_feat.shape
    # --------------------------------------------------------------------------------- #
    print(f"Number of frames F: {num_frames}")
    print(f"Number of features d: {num_feats}")

    smpl_output = dataset.get_body(human_feat[sample["padding_mask"].to(bool)])
    smpl_to_mp4(smpl_output.vertices, "smpl.mp4")
    print(f"\nCaption:\n{sample['caption_raw']['caption']}")


if __name__ == "__main__":
    main()
