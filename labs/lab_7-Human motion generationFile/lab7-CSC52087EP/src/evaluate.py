from copy import deepcopy
from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.random_utils import set_random_seed

import pandas as pd
pd.set_option('display.max_columns', None)

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
    test_dataloader = DataLoader(
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

    metrics_dict = {}
    diffuser.metrics_to_compute = ["distribution", "semantic"]
    trainer.test(model=diffuser, dataloaders=test_dataloader)
    metrics_dict["tmr/fid"] = diffuser.metrics_dict["tmr/fid"]
    metrics_dict["tmr/tmr_score"] = diffuser.metrics_dict["tmr/tmr_score"]
    metrics_dict["tmr/precision"] = diffuser.metrics_dict["tmr/precision"]
    metrics_dict["tmr/recall"] = diffuser.metrics_dict["tmr/recall"]
    metrics_dict["tmr/density"] = diffuser.metrics_dict["tmr/density"]
    metrics_dict["tmr/coverage"] = diffuser.metrics_dict["tmr/coverage"]
    metrics_dict["tmr/R1"] = diffuser.metrics_dict["tmr/R1"]
    metrics_dict["tmr/R2"] = diffuser.metrics_dict["tmr/R2"]
    metrics_dict["tmr/R3"] = diffuser.metrics_dict["tmr/R3"]

    sampler_name = config.diffuser.test_sampler.name
    module_name = config.diffuser.network.module.name
    metric_path = Path("./") / f"generation_{sampler_name}_{module_name}.csv"
    metric_df = pd.DataFrame.from_dict(metrics_dict)
    metric_df.to_csv(metric_path, index=False)
    print(f"Metrics saved to {metric_path} \n")
    print(metric_df.to_string())


if __name__ == "__main__":
    main()
