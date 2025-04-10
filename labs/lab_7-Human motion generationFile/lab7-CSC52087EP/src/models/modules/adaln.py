from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.modules.base import (
    BaseDiT,
    CrossAttentionOp,
    FusedMLP,
    LayerNorm16Bits,
    StochatichDepth,
)

# ------------------------------------------------------------------------------------- #


class AdaLNSABlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        dim_cond: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.1,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        self.initial_ln = layer_norm(dim_qkv, eps=1e-6, elementwise_affine=False)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_qkv * 6, bias=use_biases),
        )
        # Zero init
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6, elementwise_affine=False)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(cond).chunk(6, dim=-1)
        )
        attention_output = self.sa(
            modulate_shift_and_scale(self.initial_ln(tokens), shift_msa, scale_msa),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                gate_msa.unsqueeze(1) * self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.layer_scale_2
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        else:
            tokens = tokens + gate_msa.unsqueeze(1) * self.sa_stochastic_depth(
                attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        return tokens


# ------------------------------------------------------------------------------------- #


def modulate_shift_and_scale(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # ----------------------------------------------------------------------------- #
    # Complete this part for `Code 4`
    x = (1 + scale.unsqueeze(1)) * x + shift.unsqueeze(1)
    # ----------------------------------------------------------------------------- #
    return x


class AdaLNDiT(BaseDiT):
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
        assert not (clip_sequential and cond_sequential)

    # --------------------------------------------------------------------------------- #

    def init_conds_mappings(self, num_cond_feats: int, latent_dim: int):
        self.joint_cond_projection = nn.Linear(num_cond_feats, latent_dim)

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
                AdaLNSABlock(
                    dim_qkv=latent_dim,
                    dim_cond=latent_dim,
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

        self.final_norm = layer_norm(latent_dim, eps=1e-6, elementwise_affine=False)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 2, bias=True),
        )
        # Zero init
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)

    # --------------------------------------------------------------------------------- #

    def cond_mapping(self, cond: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        c_emb = self.joint_cond_projection(cond) + t
        return c_emb

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        for block in self.backbone_module:
            x = block(x, y, mask)
        return x

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        shift, scale = self.final_adaln(y).chunk(2, dim=-1)
        x = modulate_shift_and_scale(self.final_norm(x), shift, scale)
        return self.final_linear(x)


if __name__ == "__main__":
    batch_size = 2
    num_feats = 3
    num_cond_feats = 4
    num_frames = 5

    model = AdaLNDiT(
        name="adaln",
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
