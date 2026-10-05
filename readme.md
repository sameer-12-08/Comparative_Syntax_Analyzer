# Comparative Syntax Analyzer

Theory of Computation project: a deterministic CFG parser (Engine 1) vs. CodeBERT masked-LM analysis (Engine 2),
run side by side, plus a hybrid where the neural model proposes a fix and the grammar verifies it.

## Setup and run
    pip install -r requirements.txt
    streamlit run app.py                          # dashboard (tick "Enable CodeBERT" to load Engine 2)
    python tests/test_all.py                               # sanity tests
    python -m evaluation.run_benchmark                     # Engine 1 + CPython baseline
    python -m evaluation.run_benchmark --engine2 codebert  # full comparison -> evaluation/results.md

The first CodeBERT run downloads `microsoft/codebert-base-mlm` (the checkpoint with a trained MLM head).

## Layout
- `Backend/Engine1_Deterministic_Parser/lexical_analyser.py`: regex (DFA) lexer with INDENT/DEDENT.
- `Backend/Engine1_Deterministic_Parser/syntax_analyzer.py`: BNF grammar, recursive-descent LL(1) parser, derivation trace, bracket PDA simulation.
- `Backend/Engine2_Probabilistic_Neural_Analyzer/masked_language_modeling.py`: CodeBERT scoring: token surprise, then repair candidates.
- `Backend/Engine2_Probabilistic_Neural_Analyzer/semantic_optimizer.py`: hybrid repair (neural proposes, parser verifies).
- `Backend/Engine2_Probabilistic_Neural_Analyzer/transformer.py`: BPE subword view. `fake_scorer.py`: plumbing-test oracle, not a result.
- `app.py` + `Frontend/` (`input_module.py`, `execution_controler.py`, `Dual_pane_Visualizer.py`, `Token_visualization.py`): Streamlit dashboard.
- `Backend/input_sanitizer.py`, `parallel_execution_threading.py`, `Response_aggregator.py`: orchestration.
- `evaluation/`: valid snippets + injected errors (missing colon, missing/extra ")", unterminated string), benchmark.

## Method notes
- Error types are single-edit, single-line, so the parser's line is usually exact. Expect the CFG to win detection and
  localization; the interesting results are repair and the hybrid. Report what you measure, whatever it is.
- `tau` and `gamma` (Engine 2 thresholds) are uncalibrated defaults. Tune them on your machine, watching the false-positive row.
- Hybrid fix is counted correct only if the verified edit reproduces the original code (whitespace-insensitive).
- Threads do not speed up model inference (GIL); they only keep both engines running concurrently.

## Training the neural localizers (Colab)
Open `Colab_Training.ipynb` in Google Colab (T4 GPU), upload the project zip (without `.venv`) and run all cells.
`training/` generates grammar-labelled programs, trains a GRU from scratch, fine-tunes CodeBERT as a token classifier
(token on the error line = 1) and writes `results.md` comparing both with the CFG parser on a held-out test set,
the hand-written benchmark and a nesting-depth test (trained on depth <= 3, tested up to 20).
Put the downloaded `codebert_localizer/` folder in `models/`; the dashboard then uses it to pick the error line
while zero-shot CodeBERT proposes the fix.

## Sharing the trained model (do not commit it to GitHub)
GitHub rejects files over 100 MB, and `model.safetensors` is about 500 MB. Keep `models/` in `.gitignore` and publish the folder
on Hugging Face instead (`huggingface-cli upload <user>/<repo> models/codebert_localizer .`). Then type `<user>/<repo>` in the
dashboard's "Fine-tuned model folder" box; it is downloaded automatically on first use.
