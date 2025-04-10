import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------- #


def infer(net, t, x, conds, mask, guidance_weight):
    bs = x.shape[0]

    if conds is None:
        x_out = net(t.expand(bs), x, mask=mask)[0]
        return x_out

    x_both = torch.cat([x, x])
    cond_knot = torch.zeros_like(conds)
    y_both = torch.cat([conds, cond_knot])
    mask_both = torch.cat([mask, mask])
    t_both = torch.cat([t.expand(bs), t.expand(bs)])

    out = net(t_both, x_both, y=y_both, mask=mask_both)[0]
    cond_denoised, uncond_denoised = out.chunk(2)
    x_out = uncond_denoised + (cond_denoised - uncond_denoised) * guidance_weight

    return x_out


# ------------------------------------------------------------------------------------- #


class DDIMSampler(nn.Module):
    def __init__(self, scheduler: nn.Module, num_steps: int, cfg_rate: float, **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.cfg_rate = cfg_rate

    def sample(
        self,
        net,
        latents: torch.Tensor,
        conds: torch.Tensor = None,
        mask: torch.Tensor = None,
        randn_like=torch.randn_like,
    ):
        # Time step discretization
        step_indices = torch.arange(self.num_steps + 1, device=latents.device)
        t_steps = 1 - step_indices / self.num_steps
        gammas = self.scheduler(t_steps)

        # Main sampling loop
        bool_mask = ~mask.to(bool)
        x_cur = latents
        for step, (g_cur, g_next) in enumerate(zip(gammas[:-1], gammas[1:])):
            x0 = infer(net, g_cur, x_cur, conds, bool_mask, self.cfg_rate)

            # x0 prediction
            # ------------------------------------------------------------------------- #
            # Complete this part for `Code 8`
            noise_pred = (x_cur - torch.sqrt(g_cur) * x0) / torch.sqrt(1.0 - g_cur)
            x_next = torch.sqrt(g_next) * x0 + torch.sqrt(1.0 - g_next) * noise_pred
            # ------------------------------------------------------------------------- #

            x_cur = x_next

        return x_cur
