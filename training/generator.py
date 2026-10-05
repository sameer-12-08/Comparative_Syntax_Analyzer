"""Grammar-based generator of valid MiniPy programs (names defined, well-typed) + labelled broken variants.
Every program is checked by Engine 1 (syntax, names, types), so labels are exact."""
import random
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import analyze
from evaluation.dataset import INJECTORS, TYPES

INTS = ["a", "b", "c", "x", "y", "n", "i", "total", "count"]
STRS = ["name", "msg", "text", "word", "label"]
WORDS = ["hi", "ok", "done", "sam", "big", "small", "x1"]
OPS, CMPS = ["+", "-", "*"], ["<", ">", "==", "!=", "<=", ">="]


class Gen:
    def __init__(s, rng, depth=3):
        s.r, s.D, s.ints, s.strs, s.fns = rng, depth, set(), set(), {}

    def num(s): return str(s.r.randint(0, 99))

    def int_expr(s, d):
        r, c = s.r, s.r.random()
        if d <= 0 or c < 0.35:
            return r.choice(sorted(s.ints)) if s.ints and r.random() < 0.6 else s.num()
        if c < 0.75: return f"({s.int_expr(d - 1)} {r.choice(OPS)} {s.int_expr(d - 1)})"
        if c < 0.85: return f"len({s.str_expr(d - 1)})"
        if s.fns:
            f, ar = r.choice(sorted(s.fns.items()))
            return f"{f}({', '.join(s.int_expr(d - 1) for _ in range(ar))})"
        return s.num()

    def str_expr(s, d):
        r = s.r
        if d <= 0 or r.random() < 0.5:
            return r.choice(sorted(s.strs)) if s.strs and r.random() < 0.5 else '"' + r.choice(WORDS) + '"'
        return f"({s.str_expr(d - 1)} + {s.str_expr(d - 1)})"

    def block(s, depth, ind):
        return [l for _ in range(s.r.randint(1, 2)) for l in s.stmt(depth, ind)]

    def stmt(s, depth, ind):
        r, pad = s.r, "    " * ind
        kinds = ["assign", "assign", "print", "print"] + (["if", "while"] if depth > 0 else []) + (["def"] if ind == 0 and depth > 0 else [])
        k = r.choice(kinds)
        if k == "assign":
            if r.random() < 0.7:
                v = r.choice(INTS); e = s.int_expr(s.D); s.ints.add(v)
            else:
                v = r.choice(STRS); e = s.str_expr(2); s.strs.add(v)
            return [f"{pad}{v} = {e}"]
        if k == "print":
            return [f"{pad}print({s.int_expr(s.D) if r.random() < 0.6 else s.str_expr(2)})"]
        if k == "if":
            out = [f"{pad}if {s.int_expr(1)} {r.choice(CMPS)} {s.int_expr(1)}:"] + s.block(depth - 1, ind + 1)
            if r.random() < 0.4: out += [f"{pad}else:"] + s.block(depth - 1, ind + 1)
            return out
        if k == "while":
            if not s.ints: return s.stmt(0, ind)
            v = r.choice(sorted(s.ints))
            return [f"{pad}while {v} < {r.randint(3, 20)}:"] + s.block(depth - 1, ind + 1) + [f"{pad}    {v} = {v} + 1"]
        name, ar = f"f{len(s.fns)}", r.randint(1, 2)  # def
        saved = set(s.ints); s.ints |= set(["p", "q"][:ar])
        body = f"{pad}    return {s.int_expr(2)}"
        s.ints = saved; s.fns[name] = ar
        return [f"{pad}def {name}({', '.join(['p', 'q'][:ar])}):", body]


def clean(text):
    r = analyze(text)
    return r["valid"] and not r["semantic"]["issues"] and not r["types"]["errors"]


def valid_program(rng, depth=3):
    while True:
        g = Gen(rng, depth)
        lines = [l for _ in range(rng.randint(2, 6)) for l in g.stmt(2, 0)]
        lines.append(f"print({g.int_expr(1)})")
        text = "\n".join(lines)
        if len(text) <= 480 and clean(text): return text


def make_split(n, seed, depth=3, exclude=()):
    """n samples, half valid / half broken (injected edit, line known), no duplicates, none in `exclude`."""
    rng, seen, out = random.Random(seed), set(exclude), []
    while len(out) < n:
        base = valid_program(rng, depth)
        if rng.random() < 0.5:
            sample = dict(text=base, error_line=None, label="valid")
        else:
            typ = rng.choice(TYPES); res = INJECTORS[typ](base, rng)
            if not res: continue
            broken, line = res; a = analyze(broken)
            if a["valid"] or a["error"]["line"] != line: continue
            sample = dict(text=broken, error_line=line, label=typ)
        if sample["text"] in seen: continue
        seen.add(sample["text"]); out.append(sample)
    return out


def nested_expr(D, r):
    e = str(r.randint(0, 9))
    for _ in range(D): e = f"({e} {r.choice(OPS)} {r.randint(0, 9)})"
    return e


def deep_dataset(depths, n_per, seed=11):
    """One-line programs with D nested brackets: valid, missing ')' or extra ')'. Tests unbounded nesting."""
    r, out = random.Random(seed), []
    for D in depths:
        for k in range(n_per):
            base = f"x = {nested_expr(D, r)}"
            out.append(dict(text=base, error_line=None, label="valid", depth=D))
            pos = [i for i, c in enumerate(base) if c == ")"]
            p = r.choice(pos)
            broken = base[:p] + base[p + 1:] if k % 2 == 0 else base[:p + 1] + ")" + base[p + 1:]
            out.append(dict(text=broken, error_line=1, label="deep_broken", depth=D))
    return out
