"""Download the IWSLT'15 English-Vietnamese parallel corpus.

The canonical Stanford NLP mirror (nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi)
blocks direct/automated downloads (HTTP 403). This script instead pulls the same
corpus (133,317 train / 1,268 validation / 1,268 test sentence pairs, from
Luong & Manning, IWSLT 2015) from a HuggingFace parquet mirror of the identical data.
"""

import pathlib
import urllib.request

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"

FILES = {
    "train.parquet": "https://huggingface.co/datasets/Nhan2201/iwslt-eng-vi/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation.parquet": "https://huggingface.co/datasets/Nhan2201/iwslt-eng-vi/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test.parquet": "https://huggingface.co/datasets/Nhan2201/iwslt-eng-vi/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"skip (already exists): {dest}")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> saved to {dest}")


if __name__ == "__main__":
    download()
