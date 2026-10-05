"""Hybrid repair: the neural model proposes edits, the CFG parser verifies them."""
import re


def same_code(a, b):
    norm = lambda s: re.sub(r"[ \t]+", "", s).strip()
    return norm(a) == norm(b)


def hybrid_repair(candidates, parse_fn, k=10):
    """First candidate (in model-score order) that the grammar accepts, or None."""
    for rank, c in enumerate(candidates[:k], 1):
        if parse_fn(c["fixed_text"])["valid"]:
            return {**c, "rank": rank}
    return None


def _offset(text, line, col):
    return sum(len(l) + 1 for l in text.split("\n")[:line - 1]) + col


def semantic_repair(text, sem, parse_fn, scorer=None, k=10):
    """Fix candidates for undefined names: (a) wrap in quotes (a forgotten string literal, scored by
    CodeBERT when available), (b) replace by a close known name (typo). Verified by parser + name check."""
    import difflib
    known = sem["defined"] + ["print", "len", "range", "max", "min", "abs", "sum", "int", "str", "input"]
    cands = []
    for iss in sem["issues"]:
        off = _offset(text, iss["line"], iss["col"]); name = iss["name"]; end = off + len(name)
        if scorer:
            pr = scorer.symbol_probs(text[:off] + "<mask>" + text[off:], ['"', "'"])
            score, src = max(pr.values()), "CodeBERT"
        else:
            score, src = 0.5, "rule"
        cands.append(dict(op="add quotes", name=name, line=iss["line"], score=score, source=src,
                          fixed_text=text[:off] + f'"{name}"' + text[end:]))
        for m in difflib.get_close_matches(name, known, n=2, cutoff=0.7):
            cands.append(dict(op=f"rename to {m}", name=name, line=iss["line"],
                              score=difflib.SequenceMatcher(None, name, m).ratio(), source="edit distance",
                              fixed_text=text[:off] + m + text[end:]))
    cands.sort(key=lambda c: -c["score"])
    fix = None
    for rank, c in enumerate(cands[:k], 1):
        r = parse_fn(c["fixed_text"])
        if r["valid"] and not r["semantic"]["issues"] and not r["types"]["errors"]:
            fix = {**c, "rank": rank}; break
    return cands[:k], fix


def type_repair(text, types, parse_fn, k=10):
    """Rule-based repairs for operand type clashes: str operand next to a number -> int(...),
    number next to a str with '+' -> str(...). Verified: the clash count must drop and nothing new appears."""
    cands, before = [], len(types["errors"])
    for er in types["errors"]:
        if "l" not in er: continue
        base = _offset(text, er["line"], 0)
        for side, other in (("l", "rt"), ("r", "lt")):
            t = er["lt" if side == "l" else "rt"]; a, b = er[side]; seg = text[base + a:base + b]
            if t == "str" and er[other] in ("int", "float", "bool"): wrap = "int"
            elif t in ("int", "float", "bool") and er[other] == "str" and er["op"] == "+": wrap = "str"
            else: continue
            cands.append(dict(op=f"wrap in {wrap}()", name=seg, line=er["line"], score=0.5, source="rule",
                              fixed_text=text[:base + a] + f"{wrap}({seg})" + text[base + b:]))
    cands.sort(key=lambda c: c["op"] != "wrap in str()")
    for rank, c in enumerate(cands[:k], 1):
        r = parse_fn(c["fixed_text"])
        if r["valid"] and not r["semantic"]["issues"] and len(r["types"]["errors"]) < before:
            return cands[:k], {**c, "rank": rank}
    return cands[:k], None
