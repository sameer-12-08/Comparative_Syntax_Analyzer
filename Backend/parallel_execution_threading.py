"""Run both engines concurrently. Note: Python threads do not speed up CPU-bound model inference (GIL);
this keeps the UI responsive and measures per-engine latency, nothing more."""
import time
from concurrent.futures import ThreadPoolExecutor


def _timed(fn, text):
    t = time.perf_counter()
    try:
        r = fn(text)
    except Exception as e:  # one engine failing must not take the other down
        r = {"error_msg": f"{type(e).__name__}: {e}"}
    return r, (time.perf_counter() - t) * 1000


def run_parallel(text, engine1, engine2=None):
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_timed, engine1, text)
        f2 = ex.submit(_timed, engine2, text) if engine2 else None
        r1, t1 = f1.result()
        r2, t2 = f2.result() if f2 else (None, None)
    return r1, r2, {"engine1_ms": t1, "engine2_ms": t2}
