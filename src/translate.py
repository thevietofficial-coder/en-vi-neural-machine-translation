"""Load a trained checkpoint and translate EN sentences to VI (with attention weights)."""

from __future__ import annotations

import sys

import torch

from src import config
from src.data.vocab import PAD_IDX, Vocab, tokenize
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq


class Translator:
    def __init__(
        self,
        checkpoint_path=config.BEST_MODEL_PATH,
        vocab_en_path=config.VOCAB_EN_PATH,
        vocab_vi_path=config.VOCAB_VI_PATH,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.src_vocab = Vocab.load(vocab_en_path)
        self.tgt_vocab = Vocab.load(vocab_vi_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        encoder = Encoder(
            cfg["src_vocab_size"], cfg["emb_dim"], cfg["enc_hidden_dim"],
            cfg["dec_hidden_dim"], PAD_IDX, dropout=0.0,
        )
        decoder = Decoder(
            cfg["tgt_vocab_size"], cfg["emb_dim"], cfg["enc_hidden_dim"],
            cfg["dec_hidden_dim"], PAD_IDX, dropout=0.0,
        )
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def translate(self, sentence: str, max_len: int = 50):
        tokens = tokenize(sentence)
        src_ids = self.src_vocab.encode(tokens, add_sos_eos=False)
        if not src_ids:
            return "", tokens, torch.zeros(0, 0)

        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_lens = torch.tensor([len(src_ids)], dtype=torch.long)
        src_mask = src != PAD_IDX

        output_ids, attn_matrix = self.model.translate(src, src_lens, src_mask, max_len=max_len)
        out_tokens = self.tgt_vocab.decode(output_ids, strip_special=True)
        return " ".join(out_tokens), tokens, attn_matrix


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sentence = " ".join(sys.argv[1:]) or "How are you today ?"
    translator = Translator()
    translation, src_tokens, _ = translator.translate(sentence)
    print(f"EN: {sentence}")
    print(f"VI: {translation}")
