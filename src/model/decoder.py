"""GRU decoder with Bahdanau attention, one decoding step at a time."""

from __future__ import annotations

import torch
from torch import nn

from src.model.attention import BahdanauAttention


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        pad_idx: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        enc_out_dim = enc_hidden_dim * 2
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(enc_out_dim, dec_hidden_dim)
        self.rnn = nn.GRU(emb_dim + enc_out_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + enc_out_dim + dec_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,  # (batch,)
        hidden: torch.Tensor,  # (batch, dec_hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, enc_hidden_dim*2)
        src_mask: torch.Tensor,  # (batch, src_len)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)  # (batch, 1, emb_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)

        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden_new = self.rnn(rnn_input, hidden.unsqueeze(0))

        output = output.squeeze(1)
        hidden_new = hidden_new.squeeze(0)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden_new, attn_weights
