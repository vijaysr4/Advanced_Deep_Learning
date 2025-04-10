"""
This code is adapted from https://github.com/Mathux/TMR
and https://github.com/robincourant/CLaTr
"""

from pathlib import Path
from typing import Optional

from einops import repeat
import lightning as L
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

# ------------------------------------------------------------------------------------- #


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class ACTORStyleEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, x, mask) -> Tensor:
        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]


# ------------------------------------------------------------------------------------- #


class TMR(L.LightningModule):

    def __init__(
        self,
        motion_encoder: ACTORStyleEncoder,
        text_encoder: ACTORStyleEncoder,
        vae: bool,
        sample_mean: bool,
        checkpoint_dir: str,
        device: str,
        **kwargs
    ):
        super().__init__()

        self.vae = vae
        self.sample_mean = sample_mean
        self.fact = 1.0

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder

        self.checkpoint_dir = checkpoint_dir
        self.load_checkpoints()

    def load_checkpoints(self):
        # Load motion encoder
        motion_ckpt_path = Path(self.checkpoint_dir) / "tmr-motion_encoder.pt"
        self.motion_encoder.load_state_dict(
            torch.load(motion_ckpt_path, map_location=self.device)
        )
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        # Load text encoder
        text_ckpt_path = Path(self.checkpoint_dir) / "tmr-text_encoder.pt"
        self.text_encoder.load_state_dict(
            torch.load(text_ckpt_path, map_location=self.device)
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "motion":
            return self.motion_encoder

        m_num_feats = self.motion_encoder.nfeats
        t_num_feats = self.text_encoder.nfeats

        if m_num_feats == t_num_feats:
            raise ValueError(
                "Cannot automatically find the encoder (they share the same input dim)."
            )

        num_feats = inputs["x"].shape[-1]
        if num_feats == m_num_feats:
            return self.motion_encoder
        elif num_feats == t_num_feats:
            return self.text_encoder
        else:
            raise ValueError("The inputs is not recognized.")

    def encode(
        self,
        inputs,
        modality: str = "auto",
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs["x"], inputs["mask"])
        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if self.sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors
