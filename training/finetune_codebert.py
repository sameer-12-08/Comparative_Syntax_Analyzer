"""Fine-tune microsoft/codebert-base as a token classifier (token on the error line = 1)."""
import math, random
from bisect import bisect_left


def build(tok, samples, maxlen=256):
    items = []
    for s in samples:
        enc = tok(s["text"], truncation=True, max_length=maxlen, return_offsets_mapping=True)
        nl = [i for i, c in enumerate(s["text"]) if c == "\n"]
        labels = [-100 if b <= a else int(bisect_left(nl, a) + 1 == s["error_line"]) for a, b in enc["offset_mapping"]]
        items.append((enc["input_ids"], labels))
    return items


def collate(batch, pad_id, torch):
    n = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in batch])
    att = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch])
    lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch])
    return ids, att, lab


def fine_tune(train, out_dir, epochs=2, bs=16, lr=3e-5, log=print):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForTokenClassification.from_pretrained("microsoft/codebert-base", num_labels=2).to(dev)
    data = build(tok, train)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(data) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    model.train()
    for ep in range(1, epochs + 1):
        random.shuffle(data); total = 0.0
        for b in range(0, len(data), bs):
            ids, att, lab = (t.to(dev) for t in collate(data[b:b + bs], tok.pad_token_id, torch))
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); total += loss.item()
        log(f"[CodeBERT] epoch {ep}/{epochs} loss {total / math.ceil(len(data) / bs):.4f}")
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return out_dir
