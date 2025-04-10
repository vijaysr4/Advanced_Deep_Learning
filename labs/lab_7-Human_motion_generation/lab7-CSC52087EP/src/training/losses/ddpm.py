import torch
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_batches, num_frames, num_feats, num_condfeats = None, None, None, None
Feats = TensorType["num_batches", "num_feats", "num_frames"]
Mask = TensorType["num_batches", "num_frames"]
Conds = TensorType["num_batches", "num_condfeats"]

# ------------------------------------------------------------------------------------- #


class DDPMLoss:
    def __init__(self, scheduler, num_steps, **kwargs):
        self.scheduler = scheduler
        self.num_steps = num_steps

    def __call__(
        self,
        net: torch.nn.Module,
        data: Feats,
        conds: Conds = None,
        mask: Mask = None,
    ) -> float:
        # Sample noise
        timesteps = (
            torch.randint(0, self.num_steps, (data.shape[0],), device=data.device)
            / self.num_steps
        )

        gamma = self.scheduler(timesteps)[..., None, None]
        n = torch.randn_like(data)
        # ------------------------------------------------------------------------- #
        # Complete this part for `Code 6`
        data_n = torch.sqrt(gamma) * data + torch.sqrt(1 - gamma) * n
        # Denoise
        D_yn = net(gamma, data_n, conds, mask)
        # Compute loss
        loss = torch.nn.functional.mse_loss(D_yn, n)
        # ------------------------------------------------------------------------- #

        return loss


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

    def forward(self, gamma, x, y, mask):
        return x


if __name__ == "__main__":
    loss_fn = DDPMLoss(torch.nn.Identity(), 10)

    loss = loss_fn(
        net=TestNet(),
        data=torch.ones(10, 3, 5),
        conds=torch.ones(10, 2, 5),
        mask=torch.ones(10, 5),
    )

    print("Test passed!")
