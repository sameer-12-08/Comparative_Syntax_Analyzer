"""Engine 1, phase 3: type checking of expressions ("a" + 1, len(5), -"x", calling an int ...).
Types: int, float, bool, str, list, none, function, unknown. 'unknown' (parameters, user-function results,
variables assigned different types) never triggers an error, so valid code is not falsely flagged.
Like name resolution, type rules are context-sensitive: a CFG cannot express them."""

NUMS = {"int", "float", "bool"}
U = "unknown"
CMP = ("==", "!=", "<", ">", "<=", ">=")
RET = {"len": "int", "int": "int", "float": "float", "str": "str", "input": "str", "print": "none",
       "round": "int", "range": "list", "list": "list", "sorted": "list", "bool": "bool"}
FUNCS = set(RET) | {"abs", "max", "min", "sum"}


def _num(a, b):
    return "float" if "float" in (a, b) else "int"


def binop(op, a, b):
    """Result type, or None if the operator cannot be applied to these operand types."""
    if U in (a, b): return U
    if op == "+":
        if a in NUMS and b in NUMS: return _num(a, b)
        return a if a == b and a in ("str", "list") else None
    if op in ("-", "/"):
        return ("float" if op == "/" else _num(a, b)) if a in NUMS and b in NUMS else None
    if op == "*":
        if a in NUMS and b in NUMS: return _num(a, b)
        if a in ("str", "list") and b in ("int", "bool"): return a
        if b in ("str", "list") and a in ("int", "bool"): return b
        return None
    if op == "%":
        if a == "str": return "str"
        return _num(a, b) if a in NUMS and b in NUMS else None
    if op in ("<", ">", "<=", ">="):
        return "bool" if (a in NUMS and b in NUMS) or (a == b and a in ("str", "list")) else None
    return "bool"  # == and !=


class Typer:
    """Recursive descent over tokens[lo:hi]; every method returns (type, first_token, end_token)."""
    def __init__(s, toks, lo, hi, lookup, errors):
        s.t, s.p, s.hi, s.lookup, s.err = toks, lo, hi, lookup, errors

    def kind(s):
        return s.t[s.p]["kind"] if s.p < s.hi else "END"

    def problem(s, tok, msg, **extra):
        s.err.append(dict(line=tok["line"], col=tok["col"], msg=msg, **extra))

    def bad(s, op, l, r):
        cols = lambda a: (s.t[a[1]]["col"], s.t[a[2] - 1]["col"] + len(s.t[a[2] - 1]["value"]))
        s.problem(op, f"cannot apply '{op['kind']}' to {l[0]} and {r[0]}",
                  op=op["kind"], lt=l[0], rt=r[0], l=cols(l), r=cols(r))

    def or_e(s):
        r = s.and_e()
        while s.kind() == "or":
            s.p += 1; r2 = s.and_e(); r = (r[0] if r[0] == r2[0] else U, r[1], r2[2])
        return r

    def and_e(s):
        r = s.not_e()
        while s.kind() == "and":
            s.p += 1; r2 = s.not_e(); r = (r[0] if r[0] == r2[0] else U, r[1], r2[2])
        return r

    def not_e(s):
        if s.kind() == "not":
            st = s.p; s.p += 1; r = s.not_e(); return ("bool", st, r[2])
        return s.chain(CMP, s.arith)

    def chain(s, ops, sub):
        l = sub()
        while s.kind() in ops:
            op = s.t[s.p]; s.p += 1; r = sub()
            res = binop(op["kind"], l[0], r[0])
            if res is None: s.bad(op, l, r)
            l = (res or U, l[1], r[2])
        return l

    def arith(s): return s.chain(("+", "-"), s.term)
    def term(s): return s.chain(("*", "/", "%"), s.unary)

    def unary(s):
        if s.kind() == "-":
            tok, st = s.t[s.p], s.p; s.p += 1; r = s.unary()
            if r[0] not in NUMS | {U}:
                s.problem(tok, f"cannot negate {r[0]}"); return (U, st, r[2])
            return (r[0], st, r[2])
        return s.call()

    def call(s):
        r = s.atom()
        while s.kind() == "(":
            callee = s.t[r[1]] if r[2] - r[1] == 1 and s.t[r[1]]["kind"] == "NAME" else None
            s.p += 1; args = []
            if s.kind() != ")":
                args.append(s.or_e())
                while s.kind() == ",":
                    s.p += 1; args.append(s.or_e())
            s.p += 1
            r = (s.call_type(callee, r[0], [a[0] for a in args]), r[1], s.p)
        return r

    def call_type(s, callee, ftype, ts):
        if callee and ftype not in ("function", U):
            s.problem(callee, f"'{callee['value']}' is a {ftype}, not a function"); return U
        name = callee["value"] if callee else None
        if name == "len" and ts and ts[0] not in ("str", "list", U): s.problem(callee, f"len() needs str or list, got {ts[0]}")
        if name == "abs" and ts and ts[0] not in NUMS | {U}: s.problem(callee, f"abs() needs a number, got {ts[0]}")
        if name == "range" and any(t not in ("int", "bool", U) for t in ts): s.problem(callee, "range() needs integers")
        if name == "abs": return ts[0] if ts else U
        if name in ("max", "min"): return ts[0] if len(ts) > 1 and len(set(ts)) == 1 else U
        return RET.get(name, U)

    def atom(s):
        t, st = s.t[s.p], s.p
        k = t["kind"]
        if k == "(":
            s.p += 1; r = s.or_e(); s.p += 1; return (r[0], st, s.p)
        if k == "[":
            s.p += 1
            if s.kind() != "]":
                s.or_e()
                while s.kind() == ",": s.p += 1; s.or_e()
            s.p += 1; return ("list", st, s.p)
        s.p += 1
        if k == "NUMBER": return ("float" if "." in t["value"] else "int", st, s.p)
        if k == "STRING": return ("str", st, s.p)
        if k == "NAME": return (s.lookup(t["value"]), st, s.p)
        return (U, st, s.p)


def check_types(tokens):
    errors, scopes, fn_depths, depth, pending, i, n = [], [{}], [], 0, None, 0, len(tokens)

    def lookup(name):
        for sc in reversed(scopes):
            if name in sc: return sc[name]
        if name in ("True", "False"): return "bool"
        if name == "None": return "none"
        return "function" if name in FUNCS else U

    while i < n:
        k = tokens[i]["kind"]
        if k == "INDENT":
            depth += 1
            if pending is not None: scopes.append(pending); fn_depths.append(depth); pending = None
        elif k == "DEDENT":
            if fn_depths and fn_depths[-1] == depth: scopes.pop(); fn_depths.pop()
            depth -= 1
        elif k not in ("NEWLINE", "EOF"):
            j = i
            while tokens[j]["kind"] not in ("NEWLINE", "EOF"): j += 1
            T = lambda lo, hi: Typer(tokens, lo, hi, lookup, errors).or_e()
            try:
                if k == "def":
                    scopes[-1][tokens[i + 1]["value"]] = "function"
                    pending = {t["value"]: U for t in tokens[i + 3:j] if t["kind"] == "NAME"}
                elif k in ("if", "elif", "while"): T(i + 1, j - 1)
                elif k == "return":
                    if j > i + 1: T(i + 1, j)
                elif k in ("else", "pass"): pass
                elif k == "NAME" and tokens[i + 1]["kind"] == "=":
                    t, nm, cur = T(i + 2, j)[0], tokens[i]["value"], scopes[-1]
                    cur[nm] = t if cur.get(nm, t) == t else U
                else: T(i, j)
            except IndexError:
                pass
            i = j
        i += 1
    return dict(errors=errors)
