"""Normalise user input before it reaches either engine."""
MAX_CHARS = 4000
_SPECIAL = ("<mask>", "<s>", "</s>", "<pad>", "<unk>")


def sanitize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input is empty.")
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").replace("\x00", "")
    for s in _SPECIAL:  # tokenizer control strings must not come from the user
        t = t.replace(s, "")
    if len(t) > MAX_CHARS:
        raise ValueError(f"Input too long ({len(t)} > {MAX_CHARS} characters).")
    return t.rstrip("\n") + "\n"
