import torch


class CosineScheduler:
    def __init__(
        self, start: float = 1, end: float = 0, tau: float = 1.0, clip_min: float = 1e-9
    ):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min

        self.v_start = torch.cos(torch.tensor(self.start) * torch.pi / 2) ** (
            2 * self.tau
        )
        self.v_end = torch.cos(torch.tensor(self.end) * torch.pi / 2) ** (2 * self.tau)

    def __call__(self, t: float) -> float:
        output = torch.cos(
            (t * (self.end - self.start) + self.start) * torch.pi / 2
        ) ** (2 * self.tau)
        output = (self.v_end - output) / (self.v_end - self.v_start)
        return torch.clamp(output, min=self.clip_min, max=1.0)
