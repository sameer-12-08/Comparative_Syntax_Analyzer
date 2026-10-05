"""Run a MiniPy program in a restricted subprocess and report stdout or the real runtime error.
Why this is safe: only whitelisted builtins exist at run time (no open/eval/exec/__import__), MiniPy has no
attribute access or imports, a 3 s timeout kills infinite loops, and output is truncated."""
import json, subprocess, sys, time

RUNNER = r'''
import sys, json, builtins, traceback
if sys.platform.startswith("linux"):
    try:
        import resource; resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    except Exception: pass
SAFE = {n: getattr(builtins, n) for n in ("print", "len", "range", "max", "min", "abs", "sum", "int",
        "str", "float", "input", "list", "bool", "round", "sorted")}
try:
    exec(compile(sys.argv[1], "<minipy>", "exec"), {"__builtins__": SAFE})
except BaseException as e:
    fr = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<minipy>"]
    sys.stderr.write("\n@@ERR@@" + json.dumps({"type": type(e).__name__, "msg": str(e), "line": fr[-1].lineno if fr else None}))
'''
MAX_OUT = 5000


def run_code(text, timeout=3):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER, text], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return dict(ran=True, stdout="", error=None, timed_out=True, ms=(time.perf_counter() - t0) * 1000)
    err = None
    if "@@ERR@@" in p.stderr:
        err = json.loads(p.stderr.split("@@ERR@@")[-1])
    return dict(ran=True, stdout=p.stdout[:MAX_OUT], error=err, timed_out=False, ms=(time.perf_counter() - t0) * 1000)
