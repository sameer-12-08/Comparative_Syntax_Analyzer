"""Frontend controller: sanitize -> run both engines in parallel -> aggregate -> hybrid repair."""
import os
import streamlit as st
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import analyze
from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import neural_analyze
from Backend.Engine2_Probabilistic_Neural_Analyzer.semantic_optimizer import hybrid_repair, semantic_repair, type_repair
from Backend.executor import run_code
from Frontend.config import MODEL_URL
from Backend.input_sanitizer import sanitize
from Backend.parallel_execution_threading import run_parallel
from Backend.Response_aggregator import aggregate


@st.cache_resource(show_spinner="Loading CodeBERT (first run downloads the model)...")
def get_scorer():
    from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import CodeBERTScorer
    return CodeBERTScorer()


@st.cache_resource(show_spinner="Loading fine-tuned CodeBERT (first run downloads about 500 MB)...")
def get_localizer(path, url=""):
    from Backend.Engine2_Probabilistic_Neural_Analyzer.finetuned_detector import CodeBERTLocalizer
    from Backend.Engine2_Probabilistic_Neural_Analyzer.model_fetcher import ensure_weights
    if url and os.path.isdir(path):
        ensure_weights(path, url)
    return CodeBERTLocalizer(path)


def _model_source(path):
    """A local folder, or a Hugging Face repo id like 'user/name'. None if neither."""
    if not path: return None
    if os.path.isdir(path): return path
    if path.count("/") == 1 and not path.startswith(("models/", ".", "/", "~")) and not os.path.exists(path):
        return path
    return None


def run_analysis(code, use_nn, tau, gamma, run_it=False, model_dir=""):
    """Raises ValueError if the input is rejected by the sanitizer."""
    text = sanitize(code)
    scorer = get_scorer() if use_nn else None
    loc, src = None, _model_source(model_dir) if scorer else None
    if src:
        try:
            loc = get_localizer(src, os.environ.get("MODEL_URL") or MODEL_URL)
        except Exception as e:  # missing files, no internet, wrong repo id ...
            st.warning(f"Could not load the fine-tuned model from '{src}': {e}. Using zero-shot CodeBERT instead.")
    e2 = (lambda t: neural_analyze(t, scorer, tau, gamma, loc)) if scorer else None
    r1, r2, lat = run_parallel(text, analyze, e2)
    fix = hybrid_repair(r2["candidates"], analyze) if r2 and "candidates" in r2 else None
    sem_c, sem_fix = [], None
    if r1["valid"]:
        if r1["semantic"]["issues"]:
            sem_c, sem_fix = semantic_repair(text, r1["semantic"], analyze, scorer)
        if r1["types"]["errors"]:
            tc, tf = type_repair(text, r1["types"], analyze)
            sem_c, sem_fix = sem_c + tc, sem_fix or tf
    run = None
    if run_it:
        run = run_code(text) if r1["valid"] else dict(ran=False, reason="Not executed: the code does not parse.")
    return dict(run=run, sem_cands=sem_c, sem_fix=sem_fix, text=text, r1=r1, r2=r2, lat=lat, agg=aggregate(r1, r2), fix=fix, scorer=scorer)
