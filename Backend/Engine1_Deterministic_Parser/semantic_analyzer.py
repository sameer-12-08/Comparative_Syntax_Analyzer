"""Engine 1, phase 2: name resolution (use-before-define). Runs only on syntactically valid code.
A CFG cannot express this rule: 'every name used must have been declared earlier' is context-sensitive
(like {ww}), so compilers check it in a separate semantic phase after parsing."""

BUILTINS = {"print", "len", "range", "max", "min", "abs", "sum", "int", "str", "float", "input",
            "list", "bool", "round", "sorted", "True", "False", "None"}


def check_names(tokens):
    """tokens: list of token dicts. Flat define-before-use in text order, with function-parameter scopes."""
    scopes, fn_depths, depth, pending = [set()], [], 0, None
    issues, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        k = t["kind"]
        if k == "def":
            scopes[-1].add(tokens[i + 1]["value"])
            j, params = i + 3, set()
            while tokens[j]["kind"] not in (")", "NEWLINE", "EOF"):
                if tokens[j]["kind"] == "NAME": params.add(tokens[j]["value"])
                j += 1
            pending, i = params, j
            continue
        if k == "INDENT":
            depth += 1
            if pending is not None:
                scopes.append(pending); fn_depths.append(depth); pending = None
        elif k == "DEDENT":
            if fn_depths and fn_depths[-1] == depth:
                scopes.pop(); fn_depths.pop()
            depth -= 1
        elif k == "NAME":
            starts_stmt = i == 0 or tokens[i - 1]["kind"] in ("NEWLINE", "INDENT", "DEDENT")
            if starts_stmt and tokens[i + 1]["kind"] == "=":
                scopes[-1].add(t["value"])
            elif t["value"] not in BUILTINS and not any(t["value"] in s for s in scopes):
                issues.append(dict(line=t["line"], col=t["col"], name=t["value"],
                                   msg=f"name '{t['value']}' is not defined"))
        i += 1
    defined = sorted(set().union(*scopes))
    return dict(issues=issues, defined=defined)
