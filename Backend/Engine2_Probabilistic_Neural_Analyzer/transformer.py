"""Neural tokenization view: BPE subwords as CodeBERT sees them (vs. the lexer's token classes)."""


def bpe_tokens(tokenizer, text):
    return [t.replace("\u0120", "\u00b7").replace("\u010a", "\\n") for t in tokenizer.tokenize(text)]
