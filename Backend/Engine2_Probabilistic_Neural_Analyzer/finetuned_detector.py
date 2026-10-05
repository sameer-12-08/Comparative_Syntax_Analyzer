"""Inference for the fine-tuned CodeBERT error localizer (trained on Colab, see training/).
The model labels every token 1 if it lies on the erroneous line; line score = mean token probability."""
from bisect import bisect_left


def line_means(starts, probs, text):
    """Average per-unit probabilities by source line. starts[i] = char offset of unit i."""
    nl = [i for i, c in enumerate(text) if c == "\n"]
    sums, cnt = {}, {}
    for st, p in zip(starts, probs):
        ln = bisect_left(nl, st) + 1
        sums[ln] = sums.get(ln, 0.0) + p
        cnt[ln] = cnt.get(ln, 0) + 1
    return {ln: sums[ln] / cnt[ln] for ln in sums}


class CodeBERTLocalizer:
    def __init__(self, path, threshold=0.5):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch, self.thr = torch, threshold
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).eval().to(self.dev)

    def predict(self, text):
        """-> (detected, line, confidence)"""
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=256, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.dev) for k, v in enc.items()}).logits[0]
        probs = torch.softmax(logits, -1)[:, 1].tolist()
        keep = [(o[0], p) for o, p in zip(offs, probs) if o[1] > o[0]]
        means = line_means([k[0] for k in keep], [k[1] for k in keep], text)
        if not means:
            return False, None, 0.0
        line = max(means, key=means.get)
        return means[line] >= self.thr, line, means[line]

    __call__ = predict
