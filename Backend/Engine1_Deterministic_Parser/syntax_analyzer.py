"""Engine 1 - recursive-descent (LL(1)) parser for the MiniPy grammar, plus a bracket PDA trace."""
import re
from dataclasses import asdict
from .lexical_analyser import tokenize, SyntaxErr
from .semantic_analyzer import check_names
from .type_checker import check_types

GRAMMAR_BNF = """\
program     -> stmt* EOF
stmt        -> if_stmt | while_stmt | def_stmt | simple_stmt NEWLINE
simple_stmt -> 'pass' | 'return' [expr] | NAME '=' expr | expr
if_stmt     -> 'if' expr ':' block ('elif' expr ':' block)* ['else' ':' block]
while_stmt  -> 'while' expr ':' block
def_stmt    -> 'def' NAME '(' [NAME (',' NAME)*] ')' ':' block
block       -> NEWLINE INDENT stmt+ DEDENT
expr        -> and_e ('or' and_e)*
and_e       -> not_e ('and' not_e)*
not_e       -> 'not' not_e | cmp
cmp         -> arith (('=='|'!='|'<'|'>'|'<='|'>=') arith)*
arith       -> term (('+'|'-') term)*
term        -> unary (('*'|'/'|'%') unary)*
unary       -> '-' unary | call
call        -> atom ('(' args ')')*
atom        -> NAME | NUMBER | STRING | '(' expr ')' | '[' args ']'
args        -> [expr (',' expr)*]
"""
_SHOW = {"NEWLINE": "end of line", "INDENT": "an indented block", "DEDENT": "end of block", "EOF": "end of input"}
_LIT = {"NAME", "NUMBER", "STRING"}
CMP = {"==", "!=", "<", ">", "<=", ">="}


def _show(k):
    return _SHOW.get(k) or (k if k in _LIT else f"'{k}'")


def rule(name):
    """Record each rule entry: the trace is the (partial) derivation the parser followed."""
    def deco(fn):
        def wrapper(self, *a):
            self.trace.append(f"{'  ' * self.d}{name}  (line {self.cur.line})")
            self.d += 1
            try:
                return fn(self, *a)
            finally:
                self.d -= 1
        return wrapper
    return deco


class Parser:
    def __init__(self, toks):
        self.t, self.i, self.trace, self.d = toks, 0, [], 0

    @property
    def cur(self):
        return self.t[self.i]

    def fail(self, expected):
        c = self.cur
        raise SyntaxErr(f"expected {' or '.join(_show(e) for e in expected)} but found {_show(c.kind)}",
                        c.line, c.col, expected, c.kind)

    def eat(self, kind, also=()):
        if self.cur.kind != kind:
            self.fail([kind, *also])
        self.i += 1

    @rule("program")
    def program(self):
        while self.cur.kind != "EOF":
            self.stmt()

    @rule("stmt")
    def stmt(self):
        k = self.cur.kind
        if k == "if": self.if_stmt()
        elif k == "while": self.while_stmt()
        elif k == "def": self.def_stmt()
        else:
            self.simple(); self.eat("NEWLINE")

    @rule("simple_stmt")
    def simple(self):
        k = self.cur.kind
        if k == "pass": self.eat("pass")
        elif k == "return":
            self.eat("return")
            if self.cur.kind != "NEWLINE": self.expr()
        elif k == "NAME" and self.t[self.i + 1].kind == "=":
            self.eat("NAME"); self.eat("="); self.expr()
        else: self.expr()

    @rule("if_stmt")
    def if_stmt(self):
        self.eat("if"); self.expr(); self.eat(":"); self.block()
        while self.cur.kind == "elif":
            self.eat("elif"); self.expr(); self.eat(":"); self.block()
        if self.cur.kind == "else":
            self.eat("else"); self.eat(":"); self.block()

    @rule("while_stmt")
    def while_stmt(self):
        self.eat("while"); self.expr(); self.eat(":"); self.block()

    @rule("def_stmt")
    def def_stmt(self):
        self.eat("def"); self.eat("NAME"); self.eat("(")
        if self.cur.kind == "NAME":
            self.eat("NAME")
            while self.cur.kind == ",":
                self.eat(","); self.eat("NAME")
        self.eat(")", also=[","]); self.eat(":"); self.block()

    @rule("block")
    def block(self):
        self.eat("NEWLINE"); self.eat("INDENT"); self.stmt()
        while self.cur.kind not in ("DEDENT", "EOF"): self.stmt()
        self.eat("DEDENT")

    @rule("expr")
    def expr(self):
        self.and_e()
        while self.cur.kind == "or": self.i += 1; self.and_e()

    def and_e(self):
        self.not_e()
        while self.cur.kind == "and": self.i += 1; self.not_e()

    def not_e(self):
        if self.cur.kind == "not": self.i += 1; self.not_e()
        else: self.cmp()

    def cmp(self):
        self.arith()
        while self.cur.kind in CMP: self.i += 1; self.arith()

    def arith(self):
        self.term()
        while self.cur.kind in ("+", "-"): self.i += 1; self.term()

    def term(self):
        self.unary()
        while self.cur.kind in ("*", "/", "%"): self.i += 1; self.unary()

    def unary(self):
        if self.cur.kind == "-": self.i += 1; self.unary()
        else: self.call()

    def call(self):
        self.atom()
        while self.cur.kind == "(":
            self.eat("("); self.args(")")

    @rule("args")
    def args(self, close):
        if self.cur.kind != close:
            self.expr()
            while self.cur.kind == ",":
                self.eat(","); self.expr()
        self.eat(close, also=[","])

    @rule("atom")
    def atom(self):
        k = self.cur.kind
        if k in _LIT: self.i += 1
        elif k == "(":
            self.eat("("); self.expr(); self.eat(")")
        elif k == "[":
            self.eat("["); self.args("]")
        else: self.fail(["NAME", "NUMBER", "STRING", "(", "["])


def pda_trace(text):
    """Simulate a pushdown automaton over ( ) [ ] (strings/comments ignored). Independent of the parser."""
    steps, stack, problem = [], [], None
    pair = {")": "(", "]": "["}
    for ln, raw in enumerate(text.split("\n"), 1):
        code = re.sub(r"\"[^\"\n]*\"|'[^'\n]*'", lambda m: " " * len(m.group()), raw.split("#", 1)[0])
        for col, ch in enumerate(code):
            if ch in "([":
                stack.append((ch, ln, col)); act = "push"
            elif ch in ")]":
                if not stack:
                    act = "underflow"; problem = problem or f"line {ln}: '{ch}' has no opener (stack empty)"
                elif stack[-1][0] != pair[ch]:
                    act = "mismatch"; problem = problem or f"line {ln}: '{ch}' does not match '{stack[-1][0]}' on top of stack"
                else:
                    stack.pop(); act = "pop"
            else:
                continue
            steps.append(dict(line=ln, col=col, symbol=ch, action=act, stack="".join(s[0] for s in stack)))
    if stack and not problem:
        c, l, _ = stack[-1]; problem = f"line {l}: '{c}' is never closed (stack not empty at end)"
    return dict(ok=problem is None, problem=problem, steps=steps)


def analyze(text):
    res = dict(valid=False, stage=None, error=None, tokens=[], trace=[], pda=pda_trace(text),
               semantic=dict(issues=[], defined=[]), types=dict(errors=[]))
    try:
        toks = tokenize(text)
        res["tokens"] = [asdict(t) for t in toks]
    except SyntaxErr as e:
        res.update(stage="lexer", error=e.asdict()); return res
    p = Parser(toks)
    try:
        p.program(); res.update(valid=True, stage="parser")
        res["semantic"] = check_names(res["tokens"])  # phase 2, only on syntactically valid code
        res["types"] = check_types(res["tokens"])     # phase 3
    except SyntaxErr as e:
        res.update(stage="parser", error=e.asdict())
    res["trace"] = p.trace
    return res
