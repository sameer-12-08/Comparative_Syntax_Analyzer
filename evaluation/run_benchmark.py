"""python -m evaluation.run_benchmark [--engine2 none|codebert|fake] [--tau 0.05] [--gamma 0.3]"""
import argparse, ast, json, pathlib, time
from Backend.input_sanitizer import sanitize
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import analyze
from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import neural_analyze
from Backend.Engine2_Probabilistic_Neural_Analyzer.semantic_optimizer import hybrid_repair, same_code
from .dataset import build_dataset, TYPES

HERE = pathlib.Path(__file__).parent
pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"


def cpython_line(t):
    try: ast.parse(t); return None
    except SyntaxError as e: return e.lineno or -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine2", default="none", choices=["none", "codebert", "fake"])
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.3)
    a = ap.parse_args()
    scorer = None
    if a.engine2 == "codebert":
        from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import CodeBERTScorer; scorer = CodeBERTScorer()
    elif a.engine2 == "fake":
        from Backend.Engine2_Probabilistic_Neural_Analyzer.fake_scorer import FakeScorer; scorer = FakeScorer()

    rows = []
    for s in build_dataset():
        t = sanitize(s["text"])
        t0 = time.perf_counter(); r1 = analyze(t); ms1 = (time.perf_counter() - t0) * 1000
        cp = cpython_line(t)
        row = dict(id=s["id"], label=s["label"], line=s["error_line"], ms1=ms1,
                   e1=not r1["valid"], e1_line=r1["error"]["line"] if r1["error"] else None,
                   cp=cp is not None, cp_line=cp)
        if scorer:
            t0 = time.perf_counter(); r2 = neural_analyze(t, scorer, a.tau, a.gamma); row["ms2"] = (time.perf_counter() - t0) * 1000
            c = r2["candidates"]; hy = hybrid_repair(c, analyze)
            row.update(e2=r2["detected"], e2_line=r2["predicted_line"],
                       fix1=bool(c) and same_code(c[0]["fixed_text"], s["original"]),
                       hybrid=hy is not None and same_code(hy["fixed_text"], s["original"]))
        rows.append(row)

    hdr = ["Error type", "n", "CFG detect", "CFG line", "CPython line"]
    if scorer: hdr += ["CodeBERT detect", "CodeBERT line", "CodeBERT fix@1", "Hybrid fix"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for typ in TYPES:
        R = [r for r in rows if r["label"] == typ]
        c = [typ, len(R), pct([r["e1"] for r in R]), pct([r["e1_line"] == r["line"] for r in R]), pct([r["cp_line"] == r["line"] for r in R])]
        if scorer: c += [pct([r["e2"] for r in R]), pct([r["e2_line"] == r["line"] for r in R]), pct([r["fix1"] for r in R]), pct([r["hybrid"] for r in R])]
        md.append("| " + " | ".join(map(str, c)) + " |")
    V = [r for r in rows if r["label"] == "valid"]
    c = ["valid (false-positive rate)", len(V), pct([r["e1"] for r in V]), "-", pct([r["cp"] for r in V])]
    if scorer: c += [pct([r["e2"] for r in V]), "-", "-", "-"]
    md.append("| " + " | ".join(map(str, c)) + " |")
    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    md.append(f"\nMean latency: CFG parser {mean('ms1'):.2f} ms" + (f", CodeBERT {mean('ms2'):.0f} ms" if scorer else ""))
    if a.engine2 == "fake": md.append("\n**FAKE oracle scorer - plumbing test only, not results.**")
    text = "\n".join(md); print(text)
    if a.engine2 != "fake":
        (HERE / "results.md").write_text(text + "\n"); (HERE / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
