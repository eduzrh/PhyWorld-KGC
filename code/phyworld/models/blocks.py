# -*- coding: utf-8 -*-
"""Neural building blocks shared by Stage II (Section 3.3.2)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stconv import RelationGate, SpatioTemporalConv1D


class FeatureShuffle(nn.Module):
    """Channel shuffle (ShuffleNet style), the "feature shuffling" of Eq. (13)."""

    def __init__(self, groups: int = 4):
        super().__init__()
        self.groups = max(1, int(groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P, C, T = x.shape
        g = self.groups
        if g <= 1 or C % g != 0:
            return x
        return x.view(P, g, C // g, T).transpose(1, 2).reshape(P, C, T)


class MultiScaleDilatedConv(nn.Module):
    """Eq. (11): U = sum_n pi_rq[n] * U^(n),  pi_rq = softmax(MLP(E_rel[rq]))."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4), rel_dim: int = 0):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size,
                      padding=d * (kernel_size - 1) // 2, dilation=d)
            for d in self.dilations
        ])
        self.norm = nn.LayerNorm(channels)
        rel_dim = int(rel_dim) or channels
        self.pi_mlp = nn.Sequential(
            nn.Linear(rel_dim, max(32, channels // 2)), nn.GELU(),
            nn.Linear(max(32, channels // 2), len(self.dilations)),
        )

    def forward(self, x: torch.Tensor, rel_vec: torch.Tensor) -> torch.Tensor:
        """x: (P, C, T); rel_vec: (P, d) relation embedding of each trajectory."""
        pi = torch.softmax(self.pi_mlp(rel_vec), dim=-1)          # (P, N)
        out = None
        for n, conv in enumerate(self.convs):
            un = conv(x) * pi[:, n].view(-1, 1, 1)
            out = un if out is None else out + un
        out = x + out
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return F.gelu(out)


class TemporalConvolution(nn.Module):
    """Stacked multi-scale dilated temporal convolutions over the sparse tensor.

    ``backend="stconv"`` (default) is the spatio-temporal convolution of
    Eq. (11): multi-scale depthwise dilated convolution with the
    relation-conditioned dilation mixture ``pi_{r_q}``. ``backend="dense"``
    keeps a single-scale reference variant.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4), num_layers: int = 2,
                 dropout: float = 0.1, rel_dim: int = 0, backend: str = "stconv"):
        super().__init__()
        self.backend = str(backend)
        self.in_proj = nn.Conv1d(c_in, c_out, 1)
        if self.backend == "stconv":
            self.conv = SpatioTemporalConv1D(
                c_out, c_out, kernel_size, dilations, num_layers, dropout)
            self.gate = RelationGate(rel_dim or c_out, len(dilations), dropout=dropout)
        else:
            self.layers = nn.ModuleList([
                MultiScaleDilatedConv(c_out, kernel_size, dilations, rel_dim=rel_dim)
                for _ in range(max(1, num_layers))
            ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor,
                rel_vec: torch.Tensor, ablate: bool = False,
                rule_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_proj(seq)
        if ablate:                       # w/o ST-Conv.: no temporal mixing
            return h * mask.float()
        if self.backend == "stconv":
            g = self.gate(rel_vec)
            y, m = self.conv(h, mask, g, rule_gate)
            return y * m.to(y.dtype)
        for layer in self.layers:
            h = layer(h, rel_vec)
            h = self.dropout(h) * mask.float()
        return h


class StatePooling(nn.Module):
    """Eq. (13): feature shuffle -> 1x1x1 conv -> mask-aware average pooling."""

    def __init__(self, c_out: int, shuffle_groups: int = 4):
        super().__init__()
        self.shuffle = FeatureShuffle(shuffle_groups)
        self.proj = nn.Conv1d(c_out, c_out, 1)

    def forward(self, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.shuffle(v))
        m = mask.float()
        return (h * m).sum(-1) / m.sum(-1).clamp_min(1.0)