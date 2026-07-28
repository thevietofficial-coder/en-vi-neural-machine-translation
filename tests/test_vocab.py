from src.data.vocab import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, Vocab, tokenize


def test_tokenize_unescapes_html_entities_and_lowercases():
    assert tokenize("I &apos;d like some tea .") == ["i", "'d", "like", "some", "tea", "."]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("   ") == []


def test_vocab_build_respects_min_freq():
    sentences = [["a", "b"], ["a", "c"], ["a"]]
    vocab = Vocab.build(sentences, min_freq=2, max_size=100)
    # "a" appears 3 times, "b" and "c" appear once each -> dropped
    assert "a" in vocab.stoi
    assert "b" not in vocab.stoi
    assert "c" not in vocab.stoi


def test_vocab_build_respects_max_size():
    sentences = [[f"word{i}"] * 5 for i in range(20)]  # 20 distinct tokens, each freq 5
    vocab = Vocab.build(sentences, min_freq=1, max_size=4 + 5)  # 4 special tokens + 5 words
    assert len(vocab) == 9


def test_vocab_encode_decode_roundtrip():
    vocab = Vocab.build([["hello", "world"]], min_freq=1, max_size=100)
    ids = vocab.encode(["hello", "world"], add_sos_eos=True)
    assert ids[0] == SOS_IDX
    assert ids[-1] == EOS_IDX
    decoded = vocab.decode(ids, strip_special=True)
    assert decoded == ["hello", "world"]


def test_vocab_encode_unknown_token_maps_to_unk():
    vocab = Vocab.build([["hello"]], min_freq=1, max_size=100)
    ids = vocab.encode(["hello", "nonexistent_word"])
    assert ids[0] != UNK_IDX
    assert ids[1] == UNK_IDX


def test_special_token_indices_are_fixed():
    vocab = Vocab()
    assert vocab.stoi["<pad>"] == PAD_IDX
    assert vocab.stoi["<sos>"] == SOS_IDX
    assert vocab.stoi["<eos>"] == EOS_IDX
    assert vocab.stoi["<unk>"] == UNK_IDX
