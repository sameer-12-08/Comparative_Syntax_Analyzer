"""Frontend input: sidebar settings and the source-code text area."""
import streamlit as st
from evaluation.dataset import build_dataset

DEFAULT_CODE = "def add(a, b)\n    return a + b\n\nprint(add(2, 3))"


def sidebar_settings():
    bad = [d for d in build_dataset() if d["label"] != "valid"]
    with st.sidebar:
        use_nn = st.toggle("Enable CodeBERT (Engine 2)", value=False)
        tau = st.slider("tau: token 'surprise' threshold", 0.001, 0.2, 0.05)
        gamma = st.slider("gamma: minimum repair confidence", 0.05, 0.9, 0.3)
        model_dir = st.text_input("Fine-tuned model folder (optional)", value="models/codebert_localizer")
        run_it = st.toggle("Execute the program (sandboxed)", value=True)
        pick = st.selectbox("Load a benchmark example", ["(custom)"] + [d["id"] for d in bad])
    example = next((d["text"] for d in bad if d["id"] == pick), DEFAULT_CODE)
    return use_nn, tau, gamma, pick, example, run_it, model_dir


def code_input(pick, example):
    return st.text_area("Source code (MiniPy: a small subset of Python)", example, height=200, key=f"code_{pick}")
