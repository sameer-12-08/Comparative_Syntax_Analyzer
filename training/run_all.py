"""python -m training.run_all --out models [--n-train 6000] [--epochs-bert 2] [--zero-shot] [--skip-bert] [--skip-gru]
Generates data, trains the GRU and fine-tunes CodeBERT, evaluates every model (and the CFG parser) on
(1) a held-out generated test set, (2) the hand-written benchmark, (3) nesting deeper than seen in training."""
import argparse, json, os, random
from evaluation.dataset import build_dataset
from .evaluation_utils import evaluate, parser_predict
from .generator import deep_dataset, make_split

pct = lambda x: "-" if x != x else f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models"); ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--epochs-gru", type=int, default=10); ap.add_argument("--epochs-bert", type=int, default=2)
    ap.add_argument("--skip-gru", action="store_true"); ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="reuse models already saved in --out, only evaluate")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch
    print("device:", "GPU (cuda)" if torch.cuda.is_available() else "CPU  <-- very slow: Runtime > Change runtime type > T4 GPU")

    train = make_split(a.n_train, seed=1)
    val = make_split(300, seed=2, exclude={s["text"] for s in train})
    test = make_split(500, seed=3, exclude={s["text"] for s in train + val})  # whole-program disjoint
    bench = [dict(text=d["text"], error_line=d["error_line"]) for d in build_dataset()]
    depths = [2, 4, 6, 10, 15, 20]; deep = deep_dataset(depths, 20)
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | benchmark {len(bench)} | deep {len(deep)}")

    preds = {"CFG parser (Engine 1)": parser_predict}
    if not a.skip_gru:
        from .gru_model import GRUDetector
        g, gp = GRUDetector(), os.path.join(a.out, "gru.pt")
        if a.eval_only: g.load(gp)
        else: g.fit(train, val, epochs=a.epochs_gru); g.save(gp)
        preds["GRU (from scratch)"] = g.predict
    if not a.skip_bert:
        from Backend.Engine2_Probabilistic_Neural_Analyzer.finetuned_detector import CodeBERTLocalizer
        from .finetune_codebert import fine_tune
        path = os.path.join(a.out, "codebert_localizer")
        if not a.eval_only: fine_tune(train, path, epochs=a.epochs_bert)
        loc = CodeBERTLocalizer(path); preds["CodeBERT fine-tuned"] = lambda t: loc.predict(t)[:2]
    zs = None
    if a.zero_shot:
        from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import CodeBERTScorer, neural_analyze
        sc = CodeBERTScorer()
        zs = lambda t: (lambda r: (r["detected"], r["predicted_line"]))(neural_analyze(t, sc))

    sets = {"Held-out generated test": test, "Hand-written benchmark": bench}
    results, md = {}, []
    for name, data in sets.items():
        md += [f"\n### {name} (n={len(data)})", "| Model | Detection (broken) | False positives (valid) | Correct line | Accuracy |", "|---|---|---|---|---|"]
        for m, f in {**preds, **({"CodeBERT zero-shot": zs} if zs else {})}.items():
            d = data if m != "CodeBERT zero-shot" else random.Random(0).sample(data, min(len(data), 100))
            r = results.setdefault(name, {})[m] = evaluate(f, d)
            md.append(f"| {m} | {pct(r['detection'])} | {pct(r['false_pos'])} | {pct(r['line_acc'])} | {pct(r['accuracy'])} |")
    md += ["\n### Nesting-depth test: accuracy by bracket depth (training used depth <= 3)",
           "| Model | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
    for m, f in preds.items():
        row = [pct(evaluate(f, [s for s in deep if s["depth"] == d])["accuracy"]) for d in depths]
        results.setdefault("deep", {})[m] = row; md.append(f"| {m} | " + " | ".join(row) + " |")
    text = "\n".join(md); print(text)
    open(os.path.join(a.out, "results.md"), "w").write(text + "\n")
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
