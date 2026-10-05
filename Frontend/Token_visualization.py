"""Frontend: lexer tokens (DFA token classes) vs. CodeBERT BPE subwords for the same input."""
import streamlit as st
from Backend.Engine2_Probabilistic_Neural_Analyzer.transformer import bpe_tokens


def render_token_comparison(res):
    if not res["scorer"]:
        return
    with st.expander("Tokenization: lexer tokens vs. CodeBERT BPE subwords"):
        a, b = st.columns(2)
        a.write("**Lexer (DFA) tokens**")
        a.write([t["kind"] if t["kind"] in ("NAME", "NUMBER", "STRING") else (t["value"] or t["kind"]) for t in res["r1"]["tokens"]])
        b.write("**BPE subwords**")
        b.write(bpe_tokens(res["scorer"].tok, res["text"]))
