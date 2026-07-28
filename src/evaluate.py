"""Evaluate a trained checkpoint on the IWSLT'15 test split using corpus BLEU."""

from __future__ import annotations

import json
import sys

import pandas as pd
import sacrebleu
from tqdm import tqdm

from src import config
from src.translate import Translator


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    translator = Translator()
    test_df = pd.read_parquet(config.TEST_PARQUET)

    hypotheses, references = [], []
    samples = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        en, vi_ref = row["en"], row["vi"]
        hyp, _, _ = translator.translate(en)
        hypotheses.append(hyp)
        references.append(vi_ref.lower())
        if i < 10:
            samples.append({"en": en, "reference": vi_ref, "prediction": hyp})

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"corpus BLEU on tst2013 test set (n={len(test_df)}): {bleu.score:.2f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"bleu": bleu.score, "n_test_pairs": len(test_df), "samples": samples}
    (config.REPORTS_DIR / "eval_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved reports/eval_results.json")


if __name__ == "__main__":
    main()
