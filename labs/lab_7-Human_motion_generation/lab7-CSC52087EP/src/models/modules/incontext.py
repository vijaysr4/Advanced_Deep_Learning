import torch
import torch.nn as nn
from torch import Tensor

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from src.models.modules.base import BaseDiT, LayerNorm16Bits, SelfAttentionBlock

# ------------------------------------------------------------------------------------- #

allow_ops_in_compiled_graph()

# ------------------------------------------------------------------------------------- #


class InContextDiT(BaseDiT):
    def __init__(
        self,
        name: str,
        num_feats: int,
        num_cond_feats: int,
        num_frames: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
        label_dropout: float,
        clip_sequential: bool = False,
        cond_sequential: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            name=name,
            num_feats=num_feats,
            num_cond_feats=num_cond_feats,
            num_frames=num_frames,
            latent_dim=latent_dim,
            mlp_multiplier=mlp_multiplier,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            label_dropout=label_dropout,
            clip_sequential=clip_sequential,
            cond_sequential=cond_sequential,
            device=device,
        )

    # --------------------------------------------------------------------------------- #

    def init_conds_mappings(self, num_cond_feats: int, latent_dim: int):
        self.cond_projection = nn.Linear(num_cond_feats, latent_dim)

    def init_backbone(
        self,
        num_layers: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
    ):
        self.backbone_module = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latent_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(num_layers)
            ]
        )

    def init_output_projection(self, num_feats: int, latent_dim: int):
        layer_norm = LayerNorm16Bits if self.use_layernorm16 else nn.LayerNorm

        self.final_norm = layer_norm(latent_dim, eps=1e-6)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)

    # --------------------------------------------------------------------------------- #

    def cond_mapping(self, cond: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        cond_emb = self.cond_projection(cond).unsqueeze(1)
        cond_emb = torch.cat([t.unsqueeze(1), cond_emb], dim=1)
        return cond_emb

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        bs, n_y, _ = y.shape
        mask = torch.cat([torch.ones(bs, n_y, device=y.device), mask], dim=1)
        # ----------------------------------------------------------------------------- #
        # Complete this part for `Code 3`
        x = torch.cat([y, x], dim=1)
        # ----------------------------------------------------------------------------- #
        for block in self.backbone_module:
            x = block(x, mask)
        return x

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        num_y = y.shape[1]
        # ----------------------------------------------------------------------------- #
        # Complete this part for `Code 3`
        x = x[:, num_y:, :]
        # ----------------------------------------------------------------------------- #
        return self.final_linear(self.final_norm(x))


if __name__ == "__main__":
    batch_size = 2
    num_feats = 3
    num_cond_feats = 4
    num_frames = 5

    model = InContextDiT(
        name="incontext",
        num_feats=num_feats,
        num_cond_feats=num_cond_feats,
        num_frames=num_frames,
        latent_dim=16,
        mlp_multiplier=2,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        stochastic_depth=0.1,
        label_dropout=0.1,
        clip_sequential=False,
        cond_sequential=False,
        device="cuda",
    )

    x = torch.randn(batch_size, num_frames, num_feats)
    t = torch.randn(batch_size)
    y = torch.randn(batch_size, num_cond_feats)
    mask = torch.ones(batch_size, num_frames)

    out = model(x=x, timesteps=t, y=y, mask=mask)

    assert x.shape == out.shape
    print("Test passed!")
