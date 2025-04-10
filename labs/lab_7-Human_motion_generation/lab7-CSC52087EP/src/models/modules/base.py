from typing import Optional

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

batch_size, num_cond_feats = None, None

# ------------------------------------------------------------------------------------- #


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


# ------------------------------------------------------------------------------------- #


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Taken from https://github.com/NVlabs/edm
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32)
        freqs = 2 * freqs / self.num_channels
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs)
        out = torch.cat([x.cos(), x.sin()], dim=1)
        return out.to(x.dtype)


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        time_scaling: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.encode_time = PositionalEmbedding(num_channels=dim, endpoint=True)

        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t: Tensor) -> Tensor:
        time = self.encode_time(t * self.time_scaling)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time(time)


class LayerNorm16Bits(torch.nn.LayerNorm):
    """
    16-bit friendly version of torch.nn.LayerNorm
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class CrossAttentionOp(nn.Module):
    def __init__(
        self, attention_dim, num_heads, dim_q, dim_kv, use_biases=True, is_sa=False
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.use_biases = use_biases
        self.is_sa = is_sa
        if self.is_sa:
            self.qkv = nn.Linear(dim_q, attention_dim * 3, bias=use_biases)
        else:
            self.q = nn.Linear(dim_q, attention_dim, bias=use_biases)
            self.kv = nn.Linear(dim_kv, attention_dim * 2, bias=use_biases)
        self.out = nn.Linear(attention_dim, dim_q, bias=use_biases)

    def forward(self, x_to, x_from=None, attention_mask=None):
        if x_from is None:
            x_from = x_to
        if self.is_sa:
            q, k, v = self.qkv(x_to).chunk(3, dim=-1)
        else:
            q = self.q(x_to)
            k, v = self.kv(x_from).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x


class FusedMLP(nn.Sequential):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__(
            nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias),
        )


class StochatichDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.survival_prob = 1.0 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.survival_prob < 1:
            mask = (
                torch.empty(x.shape[0], 1, 1, device=x.device).uniform_()
                + self.survival_prob
            )
            mask = mask.floor()
            if self.survival_prob > 0:
                mask = mask / self.survival_prob
            return x * mask
        else:
            return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
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


class BaseDiT(nn.Module):
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
        super().__init__()
        self.name = name
        self.label_dropout = label_dropout
        self.num_feats = num_feats
        self.num_frames = num_frames
        self.clip_sequential = clip_sequential
        self.cond_sequential = cond_sequential
        self.use_layernorm16 = device == "cuda"

        self.input_projection = nn.Sequential(
            nn.Linear(num_feats, latent_dim),
            PositionalEncoding(latent_dim),
        )
        self.time_embedding = TimeEmbedder(latent_dim // 4, time_scaling=1000)
        self.init_conds_mappings(num_cond_feats, latent_dim)
        self.init_backbone(
            num_layers, latent_dim, mlp_multiplier, num_heads, dropout, stochastic_depth
        )
        self.init_output_projection(num_feats, latent_dim)

    # --------------------------------------------------------------------------------- #

    def init_conds_mappings(self, num_cond_feats, latent_dim):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def init_backbone(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def init_output_projection(self, num_feats, latent_dim):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    # --------------------------------------------------------------------------------- #

    def mask_cond(
        self, cond: TensorType["batch_size", "num_cond_feats"]
    ) -> TensorType["batch_size", "num_cond_feats"]:
        bs = cond.shape[0]
        if self.training and self.label_dropout > 0.0:
            # 1-> use null_cond, 0-> use real cond
            prob = torch.ones(bs, device=cond.device) * self.label_dropout
            if self.cond_sequential:
                mask = torch.bernoulli(prob)[:, None, None]
            else:
                mask = torch.bernoulli(prob)[:, None]
            masked_cond = cond * (1.0 - mask)
            return masked_cond
        else:
            return cond

    def cond_mapping(self, cond: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    # --------------------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        y: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        mask = mask.logical_not() if mask is not None else None
        x = self.input_projection(x)
        t = self.time_embedding(timesteps)
        if y is not None:
            y = self.mask_cond(y)
            y = self.cond_mapping(y, mask, t)
        x = self.backbone(x, y, mask)
        x = self.output_projection(x, y)
        return x
