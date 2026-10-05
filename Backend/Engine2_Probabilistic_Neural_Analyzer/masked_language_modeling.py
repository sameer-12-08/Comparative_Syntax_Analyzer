"""Engine 2 - CodeBERT masked-language-model analysis.

Stage 1 (screen): mask each token in turn; P(original token | rest) is the model's "surprise".
Stage 2 (repair): on the suspicious line, put <mask> after every token and read the probability of
syntax symbols (: ) ( ] [ , quotes); also score deleting brackets by 1 - P(token | context).
Detection = a surprising token AND a confident repair. Thresholds tau/gamma must be calibrated.
"""
import re
from bisect import bisect_right

SYMBOLS = [":", ")", "(", "]", "[", ",", '"', "'"]


class CodeBERTScorer:
    """Thin wrapper around microsoft/codebert-base-mlm (the checkpoint that has a trained MLM head). Uses the GPU if present."""

    def __init__(self, name="microsoft/codebert-base-mlm"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForMaskedLM.from_pretrained(name).eval().to(self.dev)

    def token_probs(self, text, batch=64):
        """[(char_start, char_end, P(original token | all other tokens))] for every real token."""
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=256)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        pos, out = list(range(1, len(ids) - 1)), []
        for b in range(0, len(pos), batch):
            chunk = pos[b:b + batch]
            x = torch.tensor(ids).repeat(len(chunk), 1)
            for r, i in enumerate(chunk):
                x[r, i] = self.tok.mask_token_id
            with torch.no_grad():
                logits = self.model(input_ids=x.to(self.dev)).logits
            rows = torch.arange(len(chunk), device=self.dev)
            cols = torch.tensor(chunk, device=self.dev)
            true = torch.tensor([ids[i] for i in chunk], device=self.dev)
            pr = torch.softmax(logits[rows, cols], -1)[rows, true].tolist()
            out += [(offs[i][0], offs[i][1], p) for i, p in zip(chunk, pr)]
        return out

    def symbol_probs_many(self, masked_texts, symbols=SYMBOLS, batch=32):
        """For each text containing one <mask>: {symbol: P(symbol at the mask)}. Batched for speed."""
        torch, out = self.torch, []
        sym_ids = [[i for i in (self.tok.convert_tokens_to_ids(x) for x in (s, "\u0120" + s))
                    if i != self.tok.unk_token_id] for s in symbols]
        flat, lens = [i for ids in sym_ids for i in ids], [len(ids) for ids in sym_ids]
        for b in range(0, len(masked_texts), batch):
            enc = self.tok(masked_texts[b:b + batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(self.dev)
            with torch.no_grad():
                logits = self.model(**enc).logits
            for r in range(logits.shape[0]):
                m = (enc["input_ids"][r] == self.tok.mask_token_id).nonzero()
                if len(m) == 0:
                    out.append({s: 0.0 for s in symbols}); continue
                v = torch.softmax(logits[r, m[0, 0]], -1)[flat].tolist()
                d, k = {}, 0
                for s, l in zip(symbols, lens):
                    d[s] = max(v[k:k + l], default=0.0); k += l
                out.append(d)
        return out

    def symbol_probs(self, masked_text, symbols=SYMBOLS):
        return self.symbol_probs_many([masked_text], symbols)[0]


def neural_analyze(text, scorer, tau=0.05, gamma=0.3, localizer=None):
    """localizer (optional): callable text -> (detected, line, confidence), e.g. the fine-tuned CodeBERT."""
    lines = text.split("\n")
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l) + 1)
    line_of = lambda off: bisect_right(starts, off)

    toks = [(a, b, p, text[a:b]) for a, b, p in scorer.token_probs(text)
            if b > a and (text[a:b].strip() or "\n" in text[a:b])]
    out = dict(detected=False, predicted_line=None, candidates=[], min_prob=None, most_surprising=[])
    if not toks:
        return out
    ranked = sorted(toks, key=lambda t: t[2])
    out["min_prob"] = ranked[0][2]
    out["most_surprising"] = [dict(token=t[3].replace("\n", "\\n"), line=line_of(t[0]), prob=t[2]) for t in ranked[:5]]
    if localizer is not None:  # fine-tuned model decides WHERE; the MLM below proposes WHAT to change
        det, suspect, conf = localizer(text)
        out["note"] = f"fine-tuned localizer confidence {conf:.2f}"
        if not det:
            return out
    else:
        if ranked[0][2] >= tau:  # nothing surprising: stop here (cheap path for valid code)
            return out
        suspect = line_of(ranked[0][0])
    cands = []
    for ln in sorted({suspect, suspect - 1} - {0}):
        base, line = starts[ln - 1], lines[ln - 1]
        gaps = [(m.end(), base + m.end()) for m in re.finditer(r"\w+|[^\w\s]", line)]  # after each token
        many = scorer.symbol_probs_many([text[:o] + "<mask>" + text[o:] for _, o in gaps])  # one batched call
        for (col, o), pr in zip(gaps, many):
            for sym, p in pr.items():
                cands.append(dict(op="insert", symbol=sym, line=ln, col=col, score=p,
                                  fixed_text=text[:o] + sym + text[o:]))
        for m in re.finditer(r"[()\[\]]", line):  # deletion of a bracket the model finds unlikely
            a = base + m.start()
            p = next((t[2] for t in toks if t[0] == a), None)
            if p is not None:
                cands.append(dict(op="delete", symbol=m.group(), line=ln, col=m.start(), score=1 - p,
                                  fixed_text=text[:a] + text[a + 1:]))
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if c["fixed_text"] not in seen:
            seen.add(c["fixed_text"]); uniq.append(c)
    out["candidates"] = uniq[:10]
    if localizer is not None:
        out.update(detected=True, predicted_line=suspect)
    elif uniq and uniq[0]["score"] >= gamma:
        out.update(detected=True, predicted_line=uniq[0]["line"])
    return out
