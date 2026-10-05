"""predict(text) -> (detected: bool, line: int | None). Metrics on labelled samples."""
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import analyze


def parser_predict(text):
    r = analyze(text)
    return (not r["valid"] and r["error"] is not None), (r["error"]["line"] if r["error"] else None)


def evaluate(predict, samples):
    broken = [s for s in samples if s["error_line"] is not None]
    valid = [s for s in samples if s["error_line"] is None]
    pb = [predict(s["text"]) for s in broken]; pv = [predict(s["text"]) for s in valid]
    rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    tp, tn = sum(d for d, _ in pb), sum(not d for d, _ in pv)
    return dict(n=len(samples), detection=rate([d for d, _ in pb]), false_pos=rate([d for d, _ in pv]),
                line_acc=rate([l == s["error_line"] for (d, l), s in zip(pb, broken)]),
                accuracy=(tp + tn) / len(samples))
