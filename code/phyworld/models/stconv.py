# -*- coding: utf-8 -*-
"""Spatio-temporal convolution of Stage II (Eq. (11)).

Section 3.3 convolves the rule-gated world state along the time axis.  For a
trajectory ``x`` of length ``T`` the layer applies ``M`` depthwise dilated
convolutions with dilation rates ``d_1 ... d_M`` and mixes them with the
relation-conditioned weights ``pi_{r_q} = softmax(MLP(E_rel[r_q]))``::

    y = sum_m pi_{r_q}[m] * (W_m  *_depthwise  x)        (Eq. 11)

SAME padding keeps the trajectory length ``T``; the validity mask is propagated
through exactly the same receptive field, so padded positions never leak into a
real one.  Stacking ``L_c`` layers and projecting the channels with a
``1 x 1 x 1`` convolution gives the feature-mixing output of Eq. (13).

Rule gating (Eq. (9)-(10)) enters as a multiplicative ``rule_gate`` tensor
``Gamma`` applied to the trajectory before the first convolution; with
``rule_gate=None`` the layer is the plain convolution of Eq. (11).

Depthwise dilated convolution over a ``(pair, channel, time)`` tensor is a
standard PyTorch primitive: the layer below is built from ``nn.Conv1d`` with
``groups = C`` and SAME padding, and the validity mask is propagated by
convolving the 0/1 indicator with an all-ones kernel of the same dilation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding(kernel_size: int, dilation: int) -> int:
    """Padding that keeps the time axis at length ``T`` for odd ``kernel_size``."""
    return dilation * (kernel_size - 1) // 2


class MultiScaleTemporalConv1D(nn.Module):
    """One spatio-temporal convolution layer with a dilation mixture (Eq. 11).

    Input:
      x:    (P, C_in, T)
      mask: (P, 1,    T)  (binary)
      gate: (P, M) mixture weights for each trajectory (``pi_{r_q}``)
      rule_gate: (P, 1, T) optional multiplicative ``Gamma`` of Eq. (9)

    Output:
      x':   (P, C_in, T)
      mask':(P, 1,    T)

    SAME padding keeps the length ``T`` unchanged; the validity mask is
    propagated through the very same receptive field as the depthwise conv.
    """

    def __init__(self, c_in: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        self.c_in = int(c_in)
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        assert self.kernel_size % 2 == 1, "Use odd kernel_size for same padding"
        assert len(self.dilations) >= 1
        self.mixtures = nn.ModuleDict({
            f"scale{d}": nn.Conv1d(
                in_channels=self.c_in, out_channels=self.c_in,
                kernel_size=self.kernel_size, stride=1,
                padding=_same_padding(self.kernel_size, d), dilation=d,
                groups=self.c_in, bias=False,
            )
            for d in self.dilations
        })

    def forward(self, x: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor,
                rule_gate: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if rule_gate is not None:
            x = x * rule_gate.to(dtype=x.dtype)
        P, _, _ = x.shape
        M = len(self.dilations)
        assert gate.shape == (P, M)

        mask_f = mask.to(dtype=x.dtype)
        y_mix = torch.zeros_like(x)
        mask_out = torch.zeros_like(mask_f)
        for m, (d, conv) in enumerate(zip(self.dilations, self.mixtures.values())):
            y_mix = y_mix + gate[:, m].view(P, 1, 1).to(dtype=x.dtype) * conv(x)
            # a position stays valid if any scale sees a valid input in its window
            seen = F.conv1d(
                mask_f,
                torch.ones((1, 1, self.kernel_size), device=mask_f.device, dtype=mask_f.dtype),
                stride=1, padding=_same_padding(self.kernel_size, d), dilation=d,
            )
            mask_out = torch.maximum(mask_out, (seen > 0).to(mask_out.dtype))
        return y_mix, mask_out


class SpatioTemporalConv1D(nn.Module):
    """``L_c`` stacked spatio-temporal convolution layers over (pair, time).

    Inputs:
      x:    (P, C_in, T)
      mask: (P, 1,    T)
      gate: (P, M)

    Outputs:
      y:    (P, C_out, T)
      mask: (P, 1,     T)
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 dilations: Sequence[int], num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList([
            MultiScaleTemporalConv1D(self.c_in, kernel_size=kernel_size, dilations=dilations)
            for _ in range(self.num_layers)
        ])
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.pw = nn.Conv1d(self.c_in, self.c_out, kernel_size=1, bias=True)

    def _stack(self, x, mask, gate, rule_gate):
        for i, layer in enumerate(self.layers):
            x, mask = layer(x, mask, gate, rule_gate if i == 0 else None)
            x = self.drop(self.act(x))
        return self.drop(self.pw(x)), (mask > 0).to(dtype=mask.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor,
                rule_gate: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # depthwise convolution is run in FP32 under autocast for numerical stability
        if torch.is_autocast_enabled():
            with torch.autocast(device_type=x.device.type, enabled=False):
                y32, m32 = self._stack(
                    x.float(), mask.to(dtype=torch.float32), gate.to(dtype=torch.float32),
                    None if rule_gate is None else rule_gate.float(),
                )
            return y32.to(x.dtype), m32.to(dtype=mask.dtype)
        return self._stack(x, mask, gate, rule_gate)


class RelationGate(nn.Module):
    """``pi_{r_q} = softmax(MLP(E_rel[r_q]))`` -- the scale mixture of Eq. (11)."""

    def __init__(self, rel_dim: int, num_scales: int,
                 hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = int(hidden_dim) if hidden_dim is not None else max(64, int(rel_dim))
        self.net = nn.Sequential(
            nn.Linear(int(rel_dim), h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, int(num_scales)),
        )

    def forward(self, rel_emb: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(rel_emb), dim=-1)