"""Bahdanau (additive) attention."""

from __future__ import annotations

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    def __init__(self, enc_out_dim: int, dec_hidden_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_out_dim, dec_hidden_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, dec_hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, enc_out_dim)
        src_mask: torch.Tensor,  # (batch, src_len) bool, True = valid token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_len = encoder_outputs.size(1)
        dec_hidden_rep = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = self.v(torch.tanh(self.W_enc(encoder_outputs) + self.W_dec(dec_hidden_rep)))
        energy = energy.squeeze(-1)  # (batch, src_len)
        energy = energy.masked_fill(~src_mask, float("-inf"))

        attn_weights = torch.softmax(energy, dim=1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_out_dim)
        return context, attn_weights
