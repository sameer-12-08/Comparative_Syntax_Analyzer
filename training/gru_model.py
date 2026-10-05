"""Small bidirectional GRU, trained from scratch at character level.
Task: label every character 1 if it lies on the erroneous line, else 0 (all 0 for valid programs)."""
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from Backend.Engine2_Probabilistic_Neural_Analyzer.finetuned_detector import line_means

VOCAB = "\n" + "".join(chr(c) for c in range(32, 127))
IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = padding
MAXLEN = 512


def encode(text):
    return [IDX.get(c, IDX["?"]) for c in text[:MAXLEN]]


def char_labels(text, err_line):
    labs, ln = [], 1
    for ch in text[:MAXLEN]:
        labs.append(int(ln == err_line))
        if ch == "\n": ln += 1
    return labs


class Net(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__()
        self.e = nn.Embedding(len(VOCAB) + 1, emb, padding_idx=0)
        self.g = nn.GRU(emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.o = nn.Linear(2 * hid, 2)

    def forward(self, x):
        return self.o(self.g(self.e(x))[0])


class GRUDetector:
    def __init__(self, threshold=0.5):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, self.thr = Net().to(self.dev), threshold

    def _batch(self, samples):
        x = pad_sequence([torch.tensor(encode(s["text"])) for s in samples], batch_first=True)
        y = pad_sequence([torch.tensor(char_labels(s["text"], s["error_line"])) for s in samples],
                         batch_first=True, padding_value=-100)
        return x.to(self.dev), y.to(self.dev)

    def fit(self, train, val=None, epochs=10, bs=64, lr=2e-3, log=print):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
        for ep in range(1, epochs + 1):
            self.net.train(); data = train[:]; random.shuffle(data); total = 0.0
            for b in range(0, len(data), bs):
                x, y = self._batch(data[b:b + bs])
                loss = lossf(self.net(x).reshape(-1, 2), y.reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step(); total += loss.item()
            msg = f"[GRU] epoch {ep}/{epochs} loss {total / ((len(data) + bs - 1) // bs):.4f}"
            if val:
                ok = sum(self.predict(s["text"])[0] == (s["error_line"] is not None) for s in val[:200])
                msg += f" | val detection accuracy {ok / min(len(val), 200):.2%}"
            log(msg)

    @torch.no_grad()
    def predict(self, text):
        """-> (detected, line)"""
        self.net.eval()
        ids = torch.tensor([encode(text)]).to(self.dev)
        p = torch.softmax(self.net(ids)[0], -1)[:, 1].tolist()
        means = line_means(list(range(len(p))), p, text)
        if not means: return False, None
        line = max(means, key=means.get)
        return means[line] >= self.thr, line

    def save(self, path): torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.dev)); return self
