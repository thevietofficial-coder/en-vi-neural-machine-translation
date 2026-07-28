"""PyTorch Dataset for EN-VI sentence pairs, plus a length-aware collate function."""

from __future__ import annotations

import pathlib
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from src.data.vocab import EOS_IDX, PAD_IDX, SOS_IDX, Vocab, tokenize


class TranslationDataset(Dataset):
    """Tokenizes and numericalizes an EN-VI parquet split, dropping over-long pairs."""

    def __init__(
        self,
        parquet_path: pathlib.Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = 50,
    ):
        df = pd.read_parquet(parquet_path)
        self.pairs: list[tuple[list[int], list[int]]] = []

        for en, vi in zip(df["en"], df["vi"]):
            src_tokens = tokenize(en)
            tgt_tokens = tokenize(vi)
            if not src_tokens or not tgt_tokens:
                continue
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue
            src_ids = src_vocab.encode(src_tokens, add_sos_eos=False)
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_sos_eos=True)
            self.pairs.append((src_ids, tgt_ids))

        self.src_lengths = [len(src_ids) for src_ids, _ in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)

    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)

    src_mask = src_padded != PAD_IDX
    return src_padded, src_lens, src_mask, tgt_padded


class BucketBatchSampler(Sampler[list[int]]):
    """Groups examples of similar source length into the same batch to cut padding waste.

    Batches are formed from length-sorted indices (so each batch has near-uniform
    length), then the *order of batches* is shuffled every epoch -- keeping the
    padding efficiency of sorted batching while still shuffling training order.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [
            indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


__all__ = [
    "TranslationDataset",
    "collate_batch",
    "BucketBatchSampler",
    "SOS_IDX",
    "EOS_IDX",
    "PAD_IDX",
]
