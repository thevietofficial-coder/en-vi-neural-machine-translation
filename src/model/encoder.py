"""Bidirectional GRU encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
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
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, enc_hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor, src_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)

        packed = pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: (batch, src_len, enc_hidden_dim * 2)

        # hidden: (2, batch, enc_hidden_dim) -> forward/backward of the single GRU layer
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dec_init_hidden = torch.tanh(self.fc(hidden_cat))  # (batch, dec_hidden_dim)

        return outputs, dec_init_hidden
