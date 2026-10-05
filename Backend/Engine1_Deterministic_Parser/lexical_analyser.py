"""Engine 1 - lexer. Every token class below is a regular language (a regex denotes a DFA)."""
import re
from dataclasses import dataclass

KEYWORDS = {"if", "elif", "else", "while", "def", "return", "pass", "and", "or", "not"}
TOKEN_SPEC = [
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("NAME", r"[A-Za-z_]\w*"),
    ("STRING", r"\"[^\"\n]*\"|'[^'\n]*'"),
    ("OP", r"==|!=|<=|>=|[+\-*/%<>=]"),
    ("PUNCT", r"[()\[\],:]"),
    ("SKIP", r" +"),
    ("BAD", r"."),
]
MASTER = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int


class SyntaxErr(Exception):
    def __init__(self, msg, line, col, expected=(), found=""):
        super().__init__(msg)
        self.msg, self.line, self.col = msg, line, col
        self.expected, self.found = list(expected), found

    def asdict(self):
        return dict(msg=self.msg, line=self.line, col=self.col, expected=self.expected, found=self.found)


def tokenize(text):
    """Return tokens incl. NEWLINE/INDENT/DEDENT/EOF. Raises SyntaxErr on lexical errors."""
    toks, indents, lines = [], [0], text.split("\n")
    for ln, raw in enumerate(lines, 1):
        code = raw.split("#", 1)[0].rstrip()
        if not code:
            continue
        width = len(code) - len(code.lstrip(" "))
        if width > indents[-1]:
            indents.append(width)
            toks.append(Token("INDENT", "", ln, 0))
        else:
            while width < indents[-1]:
                indents.pop()
                toks.append(Token("DEDENT", "", ln, 0))
            if width != indents[-1]:
                raise SyntaxErr("inconsistent indentation", ln, width, [], "INDENT")
        pos = width
        while pos < len(code):
            m = MASTER.match(code, pos)
            k, v = m.lastgroup, m.group()
            if k == "BAD":
                msg = "unterminated string literal" if v in "\"'" else f"unexpected character {v!r}"
                raise SyntaxErr(msg, ln, pos, [], v)
            if k != "SKIP":
                kind = v if k in ("OP", "PUNCT") or (k == "NAME" and v in KEYWORDS) else k
                toks.append(Token(kind, v, ln, pos))
            pos = m.end()
        toks.append(Token("NEWLINE", "", ln, len(code)))
    toks += [Token("DEDENT", "", len(lines), 0) for _ in indents[1:]]
    toks.append(Token("EOF", "", len(lines), 0))
    return toks
