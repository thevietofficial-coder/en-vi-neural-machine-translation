"""Shared paths and hyperparameters."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_PARQUET = RAW_DIR / "train.parquet"
VAL_PARQUET = RAW_DIR / "validation.parquet"
TEST_PARQUET = RAW_DIR / "test.parquet"

VOCAB_EN_PATH = PROCESSED_DIR / "vocab_en.json"
VOCAB_VI_PATH = PROCESSED_DIR / "vocab_vi.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"

# Data
MAX_LEN = 30
MIN_FREQ = 2
MAX_VOCAB_SIZE = 30_000

# Model
# Kept intentionally small (vs. 512/512 in the original Luong & Manning IWSLT15
# setup) because training runs on CPU here (no CUDA build available in this
# environment) -- see README "Training notes" for the tradeoff and how to scale up.
EMB_DIM = 128
ENC_HIDDEN_DIM = 128
DEC_HIDDEN_DIM = 128
DROPOUT = 0.3

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5
N_EPOCHS = 4
SEED = 42
