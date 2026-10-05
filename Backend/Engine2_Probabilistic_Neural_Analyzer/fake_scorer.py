"""PLUMBING TEST ONLY - an oracle that peeks at Engine 1. Never report its numbers as results."""
import re
from ..Engine1_Deterministic_Parser.syntax_analyzer import analyze
from .masked_language_modeling import SYMBOLS


class FakeScorer:
    def token_probs(self, text):
        r = analyze(text)
        bad = r["error"]["line"] if r["error"] else None
        out, off = [], 0
        for ln, l in enumerate(text.split("\n"), 1):
            for m in re.finditer(r"\w+|[^\w\s]", l):
                out.append((off + m.start(), off + m.end(), 0.01 if ln == bad else 0.9))
            off += len(l) + 1
        return out

    def symbol_probs(self, masked, symbols=SYMBOLS):
        return {s: (0.9 if analyze(masked.replace("<mask>", s))["valid"] else 0.01) for s in symbols}

    def symbol_probs_many(self, masked_texts, symbols=SYMBOLS):
        return [self.symbol_probs(t, symbols) for t in masked_texts]
