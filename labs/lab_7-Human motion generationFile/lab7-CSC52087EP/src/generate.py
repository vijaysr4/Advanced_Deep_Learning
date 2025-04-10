from copy import deepcopy
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from utils.random_utils import set_random_seed
from utils.rendering_utils import smpl_to_mp4

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    set_random_seed(config.seed)

    assert config.compnode.num_gpus == 1, "Evaluation script only supports single GPU"

    trainer = instantiate(config.trainer)()
    diffuser = instantiate(config.diffuser)
    dataset = instantiate(config.dataset)

    diffuser.ema.initted = torch.empty(()).to(diffuser.ema.initted)
    diffuser.ema.step = torch.empty(()).to(diffuser.ema.initted)

    checkpoint = torch.load(config.checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "tmr" not in k}
    diffuser.load_state_dict(state_dict, strict=False)
    diffuser.tmr.load_checkpoints()

    test_dataset = deepcopy(dataset).set_split("test")
    predict_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.compnode.num_workers,
        pin_memory=True,
    )
    if hasattr(test_dataset, "get_body"):
        diffuser.get_body = test_dataset.get_body
    if hasattr(test_dataset, "normalize"):
        diffuser.normalize = test_dataset.normalize
        diffuser.unnormalize = test_dataset.unnormalize

    smpl_output, caption = trainer.predict(
        model=diffuser, dataloaders=predict_dataloader
    )[0]

    sampler_name = config.diffuser.test_sampler.name
    module_name = config.diffuser.network.module.name
    smpl_to_mp4(smpl_output.vertices, f"generation_{sampler_name}_{module_name}.mp4")
    print(f"\nCaption:\n{caption[0]}")


if __name__ == "__main__":
    main()
