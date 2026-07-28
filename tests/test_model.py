import torch

from src.data.vocab import PAD_IDX
from src.model.attention import BahdanauAttention
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq

SRC_VOCAB, TGT_VOCAB = 50, 40
EMB_DIM, ENC_HIDDEN, DEC_HIDDEN = 8, 16, 16
BATCH, SRC_LEN, TGT_LEN = 4, 7, 5


def make_batch():
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
    src_lens = torch.full((BATCH,), SRC_LEN, dtype=torch.long)
    src_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.bool)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))
    return src, src_lens, src_mask, tgt


def test_attention_weights_sum_to_one_over_valid_positions():
    attn = BahdanauAttention(enc_out_dim=ENC_HIDDEN * 2, dec_hidden_dim=DEC_HIDDEN)
    decoder_hidden = torch.randn(BATCH, DEC_HIDDEN)
    encoder_outputs = torch.randn(BATCH, SRC_LEN, ENC_HIDDEN * 2)
    mask = torch.ones(BATCH, SRC_LEN, dtype=torch.bool)
    mask[:, -2:] = False  # last two positions are padding

    _, attn_weights = attn(decoder_hidden, encoder_outputs, mask)

    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(BATCH), atol=1e-5)
    # masked-out (padding) positions should receive ~zero attention
    assert torch.allclose(attn_weights[:, -2:], torch.zeros(BATCH, 2), atol=1e-5)


def test_encoder_output_shapes():
    encoder = Encoder(SRC_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    src, src_lens, _, _ = make_batch()
    outputs, hidden = encoder(src, src_lens)
    assert outputs.shape == (BATCH, SRC_LEN, ENC_HIDDEN * 2)
    assert hidden.shape == (BATCH, DEC_HIDDEN)


def test_decoder_single_step_shapes():
    encoder = Encoder(SRC_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    decoder = Decoder(TGT_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    src, src_lens, src_mask, _ = make_batch()
    encoder_outputs, hidden = encoder(src, src_lens)

    input_token = torch.randint(1, TGT_VOCAB, (BATCH,))
    prediction, hidden_new, attn_weights = decoder(input_token, hidden, encoder_outputs, src_mask)

    assert prediction.shape == (BATCH, TGT_VOCAB)
    assert hidden_new.shape == (BATCH, DEC_HIDDEN)
    assert attn_weights.shape == (BATCH, SRC_LEN)


def test_seq2seq_forward_output_shape():
    encoder = Encoder(SRC_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    decoder = Decoder(TGT_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    model = Seq2Seq(encoder, decoder, torch.device("cpu"))
    src, src_lens, src_mask, tgt = make_batch()

    output = model(src, src_lens, src_mask, tgt, teacher_forcing_ratio=1.0)
    assert output.shape == (BATCH, TGT_LEN, TGT_VOCAB)


def test_seq2seq_translate_stops_and_returns_attention():
    encoder = Encoder(SRC_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    decoder = Decoder(TGT_VOCAB, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, PAD_IDX)
    model = Seq2Seq(encoder, decoder, torch.device("cpu"))

    src = torch.randint(1, SRC_VOCAB, (1, SRC_LEN))
    src_lens = torch.tensor([SRC_LEN])
    src_mask = torch.ones(1, SRC_LEN, dtype=torch.bool)

    output_ids, attn_matrix = model.translate(src, src_lens, src_mask, max_len=10)
    assert len(output_ids) <= 10
    assert attn_matrix.shape[1] == SRC_LEN
    assert attn_matrix.shape[0] == len(output_ids) or attn_matrix.shape[0] == len(output_ids) + 1
