"""Merge both engine outputs into one comparison record."""


def aggregate(r1, r2):
    out = {"engine1_invalid": not r1["valid"]}
    if r2 and "detected" in r2:
        l1 = r1["error"]["line"] if r1["error"] else None
        out["agree_on_validity"] = (not r1["valid"]) == r2["detected"]
        out["agree_on_line"] = l1 == r2["predicted_line"]
    return out
