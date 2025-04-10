from datetime import datetime
import functools
from omegaconf.dictconfig import DictConfig
from typing import Any, Callable, Dict, List, Tuple

from ema_pytorch import EMA
import lightning as L
import numpy as np
import torch
from torchtyping import TensorType
import torch.nn as nn

from utils.random_utils import StackedRandomGenerator
from src.callbacks.metrics import MetricCallback
from utils.rendering_utils import render_frames

# ------------------------------------------------------------------------------------- #

batch_size, num_samples = None, None
num_feats, num_frames, num_tokens = None, None, None

RawTrajectory = TensorType["num_samples", "num_frames", 4, 4]
TrajFeats = TensorType["num_samples", "num_frames", "num_feats"]
CharFeats = TensorType["num_samples", "num_frames", "num_feats"]
SeqClipFeats = TensorType["num_samples", "num_tokens", "num_feats"]
ClaTrFeats = TensorType["num_samples", "num_feats"]
TMRFeats = TensorType["num_samples", "num_feats"]
HumanMLFeats = TensorType["num_samples", "num_feats"]
TrajMask = TensorType["num_samples", "num_frames"]
ClipMask = TensorType["num_samples", "num_frames"]


# ------------------------------------------------------------------------------------- #


def get_numsamples(num_tot_samples: int, num_batches: Any, batch_size: int) -> int:
    if isinstance(num_batches, int):
        return num_batches * batch_size
    elif isinstance(num_batches, float) and num_batches == 1.0:
        return num_tot_samples
    elif isinstance(num_batches, float):
        return int(num_batches * num_tot_samples)
    else:
        raise ValueError("Invalid number of batches")


# ------------------------------------------------------------------------------------- #


class Diffuser(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss: nn.Module,
        train_sampler: Callable,
        test_sampler: Callable,
        optimizer: nn.Module,
        metric_callback: MetricCallback,
        log_wandb: bool,
        guidance_weight: float,
        ema_kwargs: DictConfig,
        tmr: nn.Module,
        sync_dist: bool,
        lr_scheduler: nn.Module = None,
        **kwargs,
    ):
        super().__init__()

        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_wandb = log_wandb
        self.sync_dist = sync_dist

        # Network and EMA
        self.net = network
        self.ema = EMA(self.net, **ema_kwargs)
        self.guidance_weight = guidance_weight
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        # Optimizer and loss
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.loss_fn = loss

        self.metric_callback = metric_callback
        self.tmr = tmr
        self.tmr.eval()

    # ---------------------------------------------------------------------------------- #

    def on_fit_start(self):
        num_samples = len(self.trainer.datamodule.train_dataset)
        num_batches = self.trainer.limit_train_batches
        batch_size = self.trainer.datamodule.train_batch_size
        num_train_samples = get_numsamples(num_samples, num_batches, batch_size)

        num_samples = len(self.trainer.datamodule.eval_dataset)
        num_batches = self.trainer.limit_val_batches
        batch_size = self.trainer.datamodule.eval_batch_size
        num_val_samples = get_numsamples(num_samples, num_batches, batch_size)

        self.metric_callback = self.metric_callback(
            device=self.device,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
        )

    def on_train_start(self):
        # ----------------------------------------------------------------------------- #
        # https://stackoverflow.com/questions/73095460/assertionerror-if-capturable-false-state-steps-should-not-be-cuda-tensors # Noqa
        if self.trainer.ckpt_path is not None and isinstance(
            self.optimizers().optimizer, torch.optim.AdamW
        ):
            self.optimizers().param_groups[0]["capturable"] = True
        # ----------------------------------------------------------------------------- #

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(params=self.net.parameters())
        self.learning_rate = optimizer.param_groups[0]["lr"]

        if self.lr_scheduler_fn is None:
            return optimizer
        else:
            total_steps = self.trainer.max_epochs * len(
                self.trainer.datamodule.train_dataloader()
            )
            scheduler = self.lr_scheduler_fn(
                optimizer=optimizer, total_steps=total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if hasattr(self, "do_optimizer_step") and not self.do_optimizer_step:
            print("Skipping optimizer step")
            closure_result = optimizer_closure()
            if closure_result is not None:
                return closure_result
            else:
                return
        else:
            return super().optimizer_step(
                epoch, batch_idx, optimizer, optimizer_closure
            )

    # ---------------------------------------------------------------------------------- #

    def on_train_epoch_start(self):
        self.get_body = self.trainer.datamodule.train_dataset.get_body
        self.normalize = self.trainer.datamodule.train_dataset.normalize
        self.unnormalize = self.trainer.datamodule.train_dataset.unnormalize

    def training_step(self, batch, batch_idx):
        data = {"data": batch["human_feat"]}
        data["conds"] = batch["caption_feat"]
        data["mask"] = ~batch["padding_mask"].to(bool)

        loss = self.loss_fn(net=self.net, **data).mean()
        self.log("train/loss", loss.item(), sync_dist=self.sync_dist)

        return loss

    def on_after_backward(self):
        self.ema.update()

    # ---------------------------------------------------------------------------------- #

    def on_validation_epoch_start(self):
        self.get_body = self.trainer.datamodule.train_dataset.get_body
        self.normalize = self.trainer.datamodule.train_dataset.normalize
        self.unnormalize = self.trainer.datamodule.train_dataset.unnormalize

        self.val_plot_samples = dict()

    def validation_step(self, batch, batch_idx):
        data = {"data": batch["human_feat"]}
        data["conds"] = batch["caption_feat"]
        data["mask"] = ~batch["padding_mask"].to(bool)

        loss = self.loss_fn(net=self.net, **data).mean()
        self.log("val/loss", loss.item(), sync_dist=self.sync_dist)

        # ----------------------------------------------------------------------------- #

        # Generate samples
        cond_data, mask = data["conds"], batch["padding_mask"]
        _, gen_data = self.sample(
            self.ema.ema_model, self.train_sampler, cond_data, mask
        )
        gen_out = {"human_feat": gen_data}

        # Compute metrics
        ref_tmr, gen_tmr, text_tmr = self.infer_tmr(
            batch["human_feat"], gen_out["human_feat"], batch["caption_raw"], mask
        )
        self.metric_callback.update_tmr_metrics(
            "val", gen_tmr, ref_tmr, text_tmr, batch["index"]
        )

        # ----------------------------------------------------------------------------- #

        # Keep sample for validation plots
        if "gen_human" not in self.val_plot_samples:
            gen_human = gen_out["human_feat"][0, mask[0].to(bool)]
            self.val_plot_samples["sample_id"] = batch["sample_id"][0]
            self.val_plot_samples["gen_human"] = self.get_body(gen_human)
            self.val_plot_samples["caption"] = batch["caption_raw"]["caption"][0]

    def on_validation_epoch_end(self):
        if not self.log_wandb:
            return

        metrics_dict = {}
        metrics_dict.update(self.metric_callback.compute_tmr_metrics("val"))

        for key, value in metrics_dict.items():
            self.log(f"val/{key}", value, sync_dist=self.sync_dist)

        # ----------------------------------------------------------------------------- #

        # Get character data for visualization
        if "gen_human" in self.val_plot_samples:
            human = self.val_plot_samples["gen_human"]
            vertices_data = human.vertices.cpu()
        else:
            vertices_data = None

        # Get textual data for visualization
        if "caption" in self.val_plot_samples:
            caption = self.val_plot_samples["caption"]
        else:
            caption = ""

        # Render frames
        frames_data = render_frames(vertices_data, 360, 360)

        self.logger.log_video(
            key="val/plots/gen_animation",
            videos=[np.transpose(frames_data, (0, 3, 1, 2))],
            step=self.global_step,
            caption=[f"[Epoch {self.current_epoch}] {caption}"],
            fps=[30],
            format=["mp4"],
        )

    # --------------------------------------------------------------------------------- #

    def on_test_start(self):
        # Avoid re-instantiating the metric callback
        if isinstance(self.metric_callback, functools.partial):
            num_test_samples = len(self.trainer.test_dataloaders.dataset)
            num_batches = self.trainer.limit_test_batches
            batch_size = self.trainer.test_dataloaders.batch_size
            num_test_samples = get_numsamples(num_test_samples, num_batches, batch_size)

            self.metric_callback = self.metric_callback(
                device=self.device, num_test_samples=num_test_samples
            )

    def on_test_epoch_start(self):
        if not hasattr(self, "metrics_to_compute"):
            self.metrics_to_compute = ["distribution", "semantic"]

    def test_step(self, batch, batch_idx):
        # Generate samples
        cond_data, mask = batch["caption_feat"], batch["padding_mask"]
        _, gen_data = self.sample(
            self.ema.ema_model, self.test_sampler, cond_data, mask
        )
        gen_out = {"human_feat": gen_data}

        if batch_idx == -1:
            return gen_out

        # Compute metrics
        ref_tmr, gen_tmr, text_tmr = self.infer_tmr(
            batch["human_feat"], gen_out["human_feat"], batch["caption_raw"], mask
        )
        self.metric_callback.update_tmr_metrics(
            "test", gen_tmr, ref_tmr, text_tmr, batch["index"]
        )

    def on_test_epoch_end(self):
        metrics_dict = {}
        metrics_dict.update(self.metric_callback.compute_tmr_metrics("test"))

        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict[k] = [v.item()]
            else:
                metrics_dict[k] = [v]

        self.metrics_dict = metrics_dict

    # --------------------------------------------------------------------------------- #

    def predict_step(self, batch, batch_idx):
        # Generate samples
        cond_data, mask = batch["caption_feat"], batch["padding_mask"]
        _, gen_data = self.sample(
            self.ema.ema_model, self.test_sampler, cond_data, mask
        )
        gen_human = gen_data[0, mask[0].to(bool)]
        smpl_output = self.get_body(gen_human)

        return smpl_output, batch["caption_raw"]["caption"]

    # --------------------------------------------------------------------------------- #

    @torch.no_grad()
    def infer_tmr(
        self,
        ref_feat: TrajFeats,
        gen_feat: TrajFeats,
        text_dict: Dict[str, Any],
        feat_mask: TrajMask,
    ) -> Tuple[TMRFeats, TMRFeats, TMRFeats]:
        ref_raw_feat = self.unnormalize(ref_feat, "feat")
        n_ref_feat = self.normalize(ref_raw_feat, "tmrrifke")
        ref_tmr = self.tmr.encode({"x": n_ref_feat, "mask": feat_mask.to(bool)})

        gen_raw_feat = self.unnormalize(gen_feat, "feat")
        n_gen_feat = self.normalize(gen_raw_feat, "tmrrifke")
        gen_tmr = self.tmr.encode({"x": n_gen_feat, "mask": feat_mask.to(bool)})

        text_feat, text_mask = text_dict["clip_seq_caption"], text_dict["clip_seq_mask"]
        text_tmr = self.tmr.encode({"x": text_feat, "mask": text_mask.to(bool)})

        return ref_tmr, gen_tmr, text_tmr

    # --------------------------------------------------------------------------------- #

    def sample(
        self,
        net: torch.nn.Module,
        sampler: Callable,
        conds: TensorType["num_samples", "num_feats"],
        mask: TensorType["num_samples"],
    ) -> Tuple[List[RawTrajectory], List[RawTrajectory]]:
        # Pick latents
        num_samples = conds.shape[0]

        seeds = torch.randint(int(-1e5), int(1e5), (num_samples,))
        rnd = StackedRandomGenerator(self.device, seeds)

        # Generate latents
        sz = [num_samples, self.net.num_frames, self.net.num_feats]
        latents = {"latents": rnd.randn_rn(sz, device=self.device)}

        # Generate trajectories.
        generations = sampler.sample(
            net, conds=conds, mask=mask, randn_like=rnd.randn_like, **latents
        )

        return latents, generations
