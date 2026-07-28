"""Streamlit demo: interactive EN -> VI translation with an attention heatmap."""

import matplotlib.pyplot as plt
import streamlit as st

from src import config
from src.translate import Translator

st.set_page_config(page_title="EN -> VI Neural Machine Translation", page_icon="🇻🇳")

st.title("English -> Vietnamese Neural Machine Translation")
st.caption(
    "BiGRU encoder + Bahdanau attention decoder, trained from scratch in PyTorch "
    "on the IWSLT'15 EN-VI corpus (133k sentence pairs)."
)


@st.cache_resource
def load_translator() -> Translator:
    return Translator()


if not config.BEST_MODEL_PATH.exists():
    st.error(
        "No trained checkpoint found at `models/best_model.pt`. "
        "Run `python -m src.train` first."
    )
    st.stop()

translator = load_translator()

default_sentence = "How are you today ?"
sentence = st.text_input("English sentence", value=default_sentence)

if st.button("Translate") or sentence:
    if sentence.strip():
        translation, src_tokens, attn_matrix = translator.translate(sentence)
        st.subheader("Translation")
        st.write(translation if translation else "(empty output)")

        if attn_matrix.numel() > 0:
            st.subheader("Attention heatmap")
            out_tokens = translation.split()
            fig, ax = plt.subplots(
                figsize=(max(4, len(src_tokens) * 0.6), max(3, len(out_tokens) * 0.5))
            )
            n_rows = attn_matrix.shape[0]
            im = ax.imshow(attn_matrix[: len(out_tokens)].numpy(), cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right")
            ax.set_yticks(range(min(n_rows, len(out_tokens))))
            ax.set_yticklabels(out_tokens[:n_rows])
            ax.set_xlabel("source (English)")
            ax.set_ylabel("generated (Vietnamese)")
            fig.colorbar(im, ax=ax, label="attention weight")
            fig.tight_layout()
            st.pyplot(fig)
