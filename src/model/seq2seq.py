"""Seq2Seq wrapper: training with teacher forcing, greedy inference for translation."""

from __future__ import annotations

import random

import torch
from torch import nn

from src.data.vocab import EOS_IDX, SOS_IDX
from src.model.decoder import Decoder
from src.model.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)

        encoder_outputs, hidden = self.encoder(src, src_lens)
        input_token = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,  # (1, src_len)
        src_lens: torch.Tensor,  # (1,)
        src_mask: torch.Tensor,  # (1, src_len)
        max_len: int = 50,
    ) -> tuple[list[int], torch.Tensor]:
        self.eval()
        encoder_outputs, hidden = self.encoder(src, src_lens)

        input_token = torch.tensor([SOS_IDX], device=self.device)
        output_ids: list[int] = []
        attn_matrix = []

        for _ in range(max_len):
            prediction, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, src_mask
            )
            top1 = prediction.argmax(1)
            token_id = top1.item()
            attn_matrix.append(attn_weights.squeeze(0).cpu())
            if token_id == EOS_IDX:
                break
            output_ids.append(token_id)
            input_token = top1

        attn_tensor = torch.stack(attn_matrix) if attn_matrix else torch.zeros(0, src.size(1))
        return output_ids, attn_tensor
