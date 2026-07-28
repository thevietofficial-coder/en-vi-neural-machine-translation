import pandas as pd
import pytest
import torch

from src.data.dataset import TranslationDataset, collate_batch
from src.data.vocab import PAD_IDX, Vocab


@pytest.fixture
def tiny_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "en": ["hello world", "this sentence is far too long for the max length limit set"],
            "vi": ["xin chao the gioi", "cau nay qua dai"],
        }
    )
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def vocabs():
    src_vocab = Vocab.build([["hello", "world"]], min_freq=1, max_size=100)
    tgt_vocab = Vocab.build([["xin", "chao", "the", "gioi"]], min_freq=1, max_size=100)
    return src_vocab, tgt_vocab


def test_dataset_drops_pairs_longer_than_max_len(tiny_parquet, vocabs):
    src_vocab, tgt_vocab = vocabs
    ds = TranslationDataset(tiny_parquet, src_vocab, tgt_vocab, max_len=5)
    assert len(ds) == 1  # only the short "hello world" pair survives


def test_dataset_target_has_sos_and_eos(tiny_parquet, vocabs):
    src_vocab, tgt_vocab = vocabs
    ds = TranslationDataset(tiny_parquet, src_vocab, tgt_vocab, max_len=50)
    _, tgt_ids = ds[0]
    assert tgt_ids[0].item() == 1  # SOS_IDX
    assert tgt_ids[-1].item() == 2  # EOS_IDX


def test_collate_batch_pads_to_longest_sequence(tiny_parquet, vocabs):
    src_vocab, tgt_vocab = vocabs
    ds = TranslationDataset(tiny_parquet, src_vocab, tgt_vocab, max_len=50)
    batch = [ds[i] for i in range(len(ds))]
    src_padded, src_lens, src_mask, tgt_padded = collate_batch(batch)

    assert src_padded.shape[0] == len(batch)
    assert src_mask.dtype == torch.bool
    # mask should be True exactly where the token isn't padding
    assert torch.equal(src_mask, src_padded != PAD_IDX)
    assert src_lens.max().item() == src_padded.shape[1]
