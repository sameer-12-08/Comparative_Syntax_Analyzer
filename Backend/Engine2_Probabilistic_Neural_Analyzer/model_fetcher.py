"""Download the fine-tuned weights (too big for git) from a GitHub Release the first time they are needed."""
import os
import shutil
import urllib.request


def ensure_weights(folder, url, name="model.safetensors", min_bytes=100_000_000):
    """Make sure folder/name exists. config.json and the tokenizer files must already be in `folder` (they are small)."""
    dest = os.path.join(folder, name)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return dest
    os.makedirs(folder, exist_ok=True)
    tmp = dest + ".part"  # download to a temp name so a half-finished file is never mistaken for the model
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if os.path.getsize(tmp) < min_bytes:
        os.remove(tmp)
        raise RuntimeError("Downloaded file is too small: check the release URL (it must point to the raw model.safetensors).")
    os.replace(tmp, dest)
    return dest
