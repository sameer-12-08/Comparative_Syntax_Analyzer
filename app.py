"""Run with:  streamlit run app.py"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import streamlit as st
from Frontend.Dual_pane_Visualizer import render_dual_pane, render_execution, render_grammar, render_verdict
from Frontend.execution_controler import run_analysis
from Frontend.input_module import code_input, sidebar_settings
from Frontend.Token_visualization import render_token_comparison

st.set_page_config(page_title="Comparative Syntax Analyzer", layout="wide")
st.title("Comparative Syntax Analyzer")
st.caption("Deterministic CFG parser (Engine 1) vs. CodeBERT masked-LM (Engine 2) on the same input")

use_nn, tau, gamma, pick, example, run_it, model_dir = sidebar_settings()
code = code_input(pick, example)

if st.button("Analyze", type="primary"):
    try:
        st.session_state["res"] = run_analysis(code, use_nn, tau, gamma, run_it, model_dir)
    except ValueError as e:
        st.session_state.pop("res", None); st.error(str(e))

res = st.session_state.get("res")  # kept across reruns so results do not vanish when a widget changes
if res:
    render_verdict(res)
    render_dual_pane(res)
    render_execution(res)
    render_token_comparison(res)
render_grammar()
