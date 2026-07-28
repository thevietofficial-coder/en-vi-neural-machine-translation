"""Train the BiGRU + Bahdanau attention EN->VI translation model."""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data.dataset import BucketBatchSampler, TranslationDataset, collate_batch
from src.data.vocab import PAD_IDX, Vocab, build_and_save_vocabs
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq

sys.stdout.reconfigure(encoding="utf-8")


def get_or_build_vocabs() -> tuple[Vocab, Vocab]:
    if config.VOCAB_EN_PATH.exists() and config.VOCAB_VI_PATH.exists():
        return Vocab.load(config.VOCAB_EN_PATH), Vocab.load(config.VOCAB_VI_PATH)
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    return build_and_save_vocabs(
        train_df, config.PROCESSED_DIR, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE
    )


def run_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    teacher_forcing_ratio: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for src, src_lens, src_mask, tgt in tqdm(loader, leave=False):
        src, src_mask, tgt = src.to(device), src_mask.to(device), tgt.to(device)

        with torch.set_grad_enabled(is_train):
            output = model(src, src_lens, src_mask, tgt, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            loss = criterion(
                output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1)
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main() -> None:
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    src_vocab, tgt_vocab = get_or_build_vocabs()
    print(f"EN vocab: {len(src_vocab)}  VI vocab: {len(tgt_vocab)}")

    train_ds = TranslationDataset(config.TRAIN_PARQUET, src_vocab, tgt_vocab, config.MAX_LEN)
    val_ds = TranslationDataset(config.VAL_PARQUET, src_vocab, tgt_vocab, config.MAX_LEN)
    print(f"train pairs: {len(train_ds)}  val pairs: {len(val_ds)}")

    train_sampler = BucketBatchSampler(train_ds.src_lengths, config.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_batch)
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    encoder = Encoder(
        len(src_vocab), config.EMB_DIM, config.ENC_HIDDEN_DIM, config.DEC_HIDDEN_DIM,
        PAD_IDX, config.DROPOUT,
    )
    decoder = Decoder(
        len(tgt_vocab), config.EMB_DIM, config.ENC_HIDDEN_DIM, config.DEC_HIDDEN_DIM,
        PAD_IDX, config.DROPOUT,
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, config.N_EPOCHS + 1):
        start = time.time()
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, config.TEACHER_FORCING_RATIO
        )
        val_loss = run_epoch(model, val_loader, criterion, device, None, teacher_forcing_ratio=0.0)
        elapsed = time.time() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"epoch {epoch:02d}/{config.N_EPOCHS} | train_loss {train_loss:.3f} "
            f"| val_loss {val_loss:.3f} | {elapsed:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "emb_dim": config.EMB_DIM,
                        "enc_hidden_dim": config.ENC_HIDDEN_DIM,
                        "dec_hidden_dim": config.DEC_HIDDEN_DIM,
                        "dropout": config.DROPOUT,
                        "src_vocab_size": len(src_vocab),
                        "tgt_vocab_size": len(tgt_vocab),
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                config.BEST_MODEL_PATH,
            )
            print(f"  -> saved new best model (val_loss={val_loss:.3f})")

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("EN->VI Seq2Seq + Attention: training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "figures" / "loss_curve.png", dpi=150)
    print("saved reports/figures/loss_curve.png")


if __name__ == "__main__":
    main()
