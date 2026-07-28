"""Tokenization and vocabulary building for the EN-VI translation pipeline.

The IWSLT'15 corpus ships Moses-tokenized (punctuation already space-separated,
HTML entities like ``&apos;`` in place of raw characters), so tokenization here
is deliberately simple: unescape HTML entities, lowercase, split on whitespace.
"""

from __future__ import annotations

import html
import json
import pathlib
from collections import Counter

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def tokenize(text: str) -> list[str]:
    text = html.unescape(text).strip().lower()
    if not text:
        return []
    return text.split()


class Vocab:
    """A simple word-level vocabulary with frequency-based pruning."""

    def __init__(self, itos: list[str] | None = None):
        self.itos: list[str] = itos if itos is not None else list(SPECIAL_TOKENS)
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, tokenized_sentences, min_freq: int = 2, max_size: int = 50_000) -> "Vocab":
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)

        itos = list(SPECIAL_TOKENS)
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(itos) >= max_size:
                break
            itos.append(token)
        return cls(itos)

    def encode(self, tokens: list[str], add_sos_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> list[str]:
        tokens = [self.itos[i] for i in ids if i < len(self.itos)]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return tokens

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> "Vocab":
        itos = json.loads(path.read_text(encoding="utf-8"))
        return cls(itos)


def build_and_save_vocabs(
    train_df,
    out_dir: pathlib.Path,
    min_freq: int = 2,
    max_size: int = 50_000,
) -> tuple[Vocab, Vocab]:
    out_dir.mkdir(parents=True, exist_ok=True)
    en_tokens = [tokenize(s) for s in train_df["en"]]
    vi_tokens = [tokenize(s) for s in train_df["vi"]]

    src_vocab = Vocab.build(en_tokens, min_freq=min_freq, max_size=max_size)
    tgt_vocab = Vocab.build(vi_tokens, min_freq=min_freq, max_size=max_size)

    src_vocab.save(out_dir / "vocab_en.json")
    tgt_vocab.save(out_dir / "vocab_vi.json")
    return src_vocab, tgt_vocab


if __name__ == "__main__":
    import pandas as pd

    root = pathlib.Path(__file__).resolve().parents[2]
    train_df = pd.read_parquet(root / "data" / "raw" / "train.parquet")
    src_vocab, tgt_vocab = build_and_save_vocabs(train_df, root / "data" / "processed")
    print(f"EN vocab size: {len(src_vocab)}")
    print(f"VI vocab size: {len(tgt_vocab)}")
