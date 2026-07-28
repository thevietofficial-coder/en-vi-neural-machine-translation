# English -> Vietnamese Neural Machine Translation

[![Tests](https://github.com/thevietofficial-coder/en-vi-neural-machine-translation/actions/workflows/tests.yml/badge.svg)](https://github.com/thevietofficial-coder/en-vi-neural-machine-translation/actions/workflows/tests.yml)

A sequence-to-sequence translator (BiGRU encoder + Bahdanau attention decoder)
trained **from scratch in PyTorch** on the real IWSLT'15 English-Vietnamese
corpus (133,317 TED-talk sentence pairs, Luong & Manning 2015).

> Built to apply the sequence-model concepts from Andrew Ng's *Sequence Models*
> (Coursera, Course 5, Weeks 1&ndash;3) &mdash; RNN/GRU encoders, the attention
> mechanism, encoder-decoder architectures &mdash; on a real bilingual corpus,
> with the model, training loop, and app written independently rather than
> reusing the course's own (much smaller, synthetic date-translation) assignment.

## Why this project

The course's Week 3 assignment applies attention to a toy task: translating
10,000 synthetic human-readable dates ("25th of June, 2009") into a fixed
format. This project keeps the same architecture family but applies it to a
**real** translation problem: 133k naturally-occurring English/Vietnamese
sentence pairs from TED talks, with real vocabulary size (~29k English /
~12.5k Vietnamese word types), variable-length sentences, and a standard
academic benchmark (BLEU) instead of exact-match accuracy.

## Architecture

```
EN sentence -> [Embedding] -> [Bidirectional GRU encoder] -> encoder_outputs (per source token)
                                                                       |
VI token t-1 -> [Embedding] --concat--> [GRU decoder] <-- context_t --[Bahdanau attention]
                                              |
                                       [Linear -> softmax] -> VI token t
```

- **Encoder**: word embeddings -> bidirectional GRU. Source sentences are
  packed with `pack_padded_sequence` so padding never affects the hidden
  state. The concatenated final forward/backward hidden state is projected
  (`tanh(W*h + b)`) into the decoder's initial hidden state.
- **Attention** (`src/model/attention.py`): additive/Bahdanau attention.
  At each decoder step, every encoder position gets a score
  `v^T * tanh(W_enc*h_enc + W_dec*h_dec)`, softmax-normalized over the
  *unpadded* source positions only (via an explicit boolean mask), producing
  a context vector as their weighted sum.
- **Decoder** (`src/model/decoder.py`): GRU driven by `[prev_token_embedding ; context]`,
  trained with teacher forcing (ratio 0.5) and, at inference, greedy decoding
  one token at a time until `<eos>` or `max_len`.
- **Bucketed batching** (`src/data/dataset.py::BucketBatchSampler`): batches
  are built from length-sorted sentences (batch order still shuffled each
  epoch) so a single very long sentence doesn't force padding onto an entire
  batch of otherwise-short ones &mdash; this alone meaningfully cuts CPU
  training time on this dataset.

## Results

Corpus BLEU on the held-out `test` split (1,268 sentence pairs):

**BLEU: 7.02** (see `reports/eval_results.json`, produced by `src/evaluate.py`)

| Metric | Value |
|---|---|
| Train pairs (<=30 tokens both sides) | 96,537 |
| Validation pairs | 861 |
| Test pairs | 1,268 |
| Trainable parameters | 12,257,253 |
| Epochs trained | 4 |
| Final train loss | 4.104 |
| Final val loss | 4.919 (still decreasing every epoch) |
| Test BLEU | 7.02 |

![Training curve](reports/figures/loss_curve.png)

Val loss was still dropping at epoch 4/4 (5.58 &rarr; 5.26 &rarr; 5.05 &rarr; 4.92) with
no sign of overfitting yet, so BLEU is capped by *training budget*, not by
the architecture &mdash; see "Training notes" below for exactly what
compute/time tradeoff produced this number and how to push it higher.

### Sample translations

Short sentences translate reasonably; longer ones expose a textbook failure
mode of greedy-decoded, undertrained seq2seq models &mdash; repetition
looping (the decoder gets "stuck" re-emitting a high-probability word because
nothing enforces that each source word be covered once):

| English | Reference (Vietnamese) | Model output |
|---|---|---|
| And I was very proud . | Tôi đã rất tự hào về đất nước tôi . | và tôi rất hào rất hào . |
| I was so shocked . | Tôi đã bị sốc . | tôi đã rất sốc . |
| My family was not poor , and myself , I had never experienced hunger . | Gia đình của tôi không nghèo , và bản thân tôi thì chưa từng phải chịu đói . | đình tôi không không nghèo , , , tôi , chưa bao giờ bao giờ . |
| In school , we spent a lot of time studying the history of Kim Il-Sung , but we never learned much about the outside world ... | Ở trường , chúng tôi dành rất nhiều thời gian để học về cuộc đời của chủ tịch Kim II-Sung ... | trường trường , chúng tôi đã nhiều nhiều về những những , , , nhưng chúng tôi không bao giờ , , , , , , ... (repetition loop) |

Full set of 10 held-out samples in `reports/eval_results.json`.

### Attention heatmap (from the Streamlit app)

The app visualizes which English words the decoder attended to while
generating each Vietnamese word &mdash; the classic sanity check for whether
attention is actually learning alignment rather than just memorizing.

## Training notes

This environment has an NVIDIA RTX 4060 GPU, but only a CPU-only PyTorch
build (`torch==2.13.0+cpu`) was available to install (the CUDA wheel index
had no matching build for the installed Python 3.14 at training time), so
the model was trained on CPU. To keep CPU training tractable, hidden/embedding
dimensions were kept modest (128 vs. the 512/512 used in the original
Luong & Manning IWSLT15 setup), sentences were capped at 30 tokens, and
training was stopped after 4 epochs (~70 minutes total on a 20-core CPU) with
a length-bucketed batch sampler to cut padding waste (see `src/config.py`
and `src/data/dataset.py::BucketBatchSampler`).

That budget lands the model at **BLEU 7.02** &mdash; well below the ~23-26
BLEU published for this exact dataset with the full-size (512-dim, GPU-trained,
many-epoch) setup, and the sample translations above show why: short sentences
are already coherent, but longer ones fall into repetition loops because (a)
greedy decoding has no mechanism to penalize re-visiting the same output word,
and (b) 4 epochs isn't enough for the attention alignment to fully sharpen.
Both are addressable without changing the architecture:

- **More epochs / a GPU build of PyTorch** &mdash; val loss was still falling
  at epoch 4/4 with no overfitting, so BLEU has clear headroom left on the
  table purely from more training.
- **Beam search instead of greedy decoding** at inference (`Seq2Seq.translate`
  currently takes the single argmax token each step).
- Raising `EMB_DIM` / `ENC_HIDDEN_DIM` / `DEC_HIDDEN_DIM` back to 512 and
  `MAX_LEN` to 50, matching the original paper's setup.

## Project structure

```
en-vi-neural-machine-translation/
├── app.py                   # Streamlit demo: EN sentence -> VI translation + attention heatmap
├── data/raw/                # train/validation/test parquet (download via src/data/download.py)
├── data/processed/          # saved vocab_en.json / vocab_vi.json
├── models/best_model.pt     # best checkpoint by validation loss
├── notebooks/01_eda.ipynb   # corpus exploration (length distributions, samples)
├── reports/                 # loss curve, BLEU + sample translations (eval_results.json)
├── src/
│   ├── config.py            # paths + hyperparameters
│   ├── data/
│   │   ├── download.py      # fetch the IWSLT'15 EN-VI corpus
│   │   ├── vocab.py         # tokenizer + Vocab (build/save/load)
│   │   └── dataset.py       # TranslationDataset, collate_fn, BucketBatchSampler
│   ├── model/
│   │   ├── encoder.py       # bidirectional GRU encoder
│   │   ├── attention.py     # Bahdanau attention
│   │   ├── decoder.py       # attention-driven GRU decoder
│   │   └── seq2seq.py       # training (teacher forcing) + greedy inference
│   ├── train.py              # training loop, checkpointing, loss curve
│   ├── evaluate.py           # corpus BLEU on the test split
│   └── translate.py          # CLI + Translator class used by app.py
└── tests/                    # vocab, dataset, and model unit tests (pytest)
```

## How to run

```bash
pip install -r requirements-dev.txt

# 1. Download the corpus (IWSLT'15 EN-VI, 133k/1.3k/1.3k train/val/test pairs)
python -m src.data.download

# 2. Train (builds and caches the vocab on first run)
python -m src.train

# 3. Evaluate BLEU on the test set
python -m src.evaluate

# 4. Try it interactively
streamlit run app.py
```

Run the test suite with:

```bash
pytest -v
```

## Dataset

[IWSLT'15 English-Vietnamese](https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/)
(Luong & Manning, *Stanford Neural Machine Translation Systems for Spoken
Language Domain*, IWSLT 2015) &mdash; 133,317 train / 1,268 validation / 1,268
test sentence pairs from TED talk transcripts. The original host blocks
automated downloads, so `src/data/download.py` pulls an identical mirror
hosted as Parquet on HuggingFace (`Nhan2201/iwslt-eng-vi`).

## License

MIT &mdash; see [LICENSE](LICENSE).
