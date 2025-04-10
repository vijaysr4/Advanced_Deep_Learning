from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.modules.base import (
    BaseDiT,
    CrossAttentionOp,
    FusedMLP,
    LayerNorm16Bits,
    PositionalEncoding,
    SelfAttentionBlock,
    StochatichDepth,
)

# ------------------------------------------------------------------------------------- #


class CrossAttentionSABlock(nn.Module):
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
        layer_scale_value: float = 0.0,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.ca = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_cond,
            is_sa=False,
            use_biases=use_biases,
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.ca_ln = layer_norm(dim_qkv, eps=1e-6)

        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
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
        cond_mask: Optional[torch.Tensor] = None,
    ):
        if cond_mask is None:
            cond_attention_mask = None
        else:
            cond_attention_mask = torch.ones(
                cond.shape[0],
                1,
                cond.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ) * token_mask.unsqueeze(2)
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

        # ----------------------------------------------------------------------------- #
        # Complete this part for `Code 5`
        normed_tokens = self.ca_ln(tokens)
        ca_output = self.ca(
            normed_tokens,
            cond,
            attention_mask=cond_attention_mask,
        )
        # ----------------------------------------------------------------------------- #
        ca_output = torch.nan_to_num(
            ca_output, nan=0.0, posinf=0.0, neginf=0.0
        )  # Needed as some tokens get attention from no token so Nan
        tokens = tokens + self.ca_stochastic_depth(ca_output)
        attention_output = self.sa(
            self.initial_ln(tokens),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens


# ------------------------------------------------------------------------------------- #


class CrossAttentionDiT(BaseDiT):
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
        clip_sequential: bool = True,
        cond_sequential: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_multiplier = mlp_multiplier
        self.stochastic_depth = stochastic_depth

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
        assert clip_sequential and cond_sequential

    # --------------------------------------------------------------------------------- #

    def init_conds_mappings(self, num_cond_feats: int, latent_dim: int):
        self.cond_projection = nn.Linear(num_cond_feats, latent_dim)

        self.cond_sa = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latent_dim,
                    num_heads=self.num_heads,
                    mlp_multiplier=self.mlp_multiplier,
                    dropout=self.dropout,
                    stochastic_depth=self.stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(2)
            ]
        )
        self.cond_positional_embedding = PositionalEncoding(latent_dim, max_len=10000)

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
                CrossAttentionSABlock(
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

        self.final_norm = layer_norm(latent_dim, eps=1e-6)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)

    # --------------------------------------------------------------------------------- #

    def cond_mapping(self, cond: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        cond_emb = self.cond_projection(cond)
        # ----------------------------------------------------------------------------- #
        # Complete this part for `Code 5`
        cond_emb = torch.cat([t.unsqueeze(1), cond_emb], dim=1)
        # ----------------------------------------------------------------------------- #
        cond_emb = self.cond_positional_embedding(cond_emb)
        # ----------------------------------------------------------------------------- #
        # Complete this part for `Code 5`
        for sa_block in self.cond_sa:
            cond_emb = sa_block(cond_emb)
        # ----------------------------------------------------------------------------- #
        return cond_emb

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        for block in self.backbone_module:
            x = block(x, y, mask, None)
        return x

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        return self.final_linear(self.final_norm(x))


if __name__ == "__main__":
    batch_size = 2
    num_feats = 3
    num_cond_feats = 4
    num_cond_temp = 6
    num_frames = 5

    model = CrossAttentionDiT(
        name="cross_attention",
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
        clip_sequential=True,
        cond_sequential=True,
        device="cuda",
    )

    x = torch.randn(batch_size, num_frames, num_feats)
    t = torch.randn(batch_size)
    y = torch.randn(batch_size, num_cond_temp, num_cond_feats)
    mask = torch.ones(batch_size, num_frames)

    out = model(x=x, timesteps=t, y=y, mask=mask)

    assert x.shape == out.shape
    print("Test passed!")
