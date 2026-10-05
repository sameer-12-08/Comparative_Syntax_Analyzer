import ast, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import analyze, pda_trace
from Backend.Engine2_Probabilistic_Neural_Analyzer.fake_scorer import FakeScorer
from Backend.Engine2_Probabilistic_Neural_Analyzer.masked_language_modeling import neural_analyze
from Backend.Engine2_Probabilistic_Neural_Analyzer.semantic_optimizer import hybrid_repair, same_code
from Backend.input_sanitizer import sanitize
from evaluation.dataset import build_dataset, load_base


def test_valid_snippets_agree_with_cpython():
    for t in load_base():
        ast.parse(t); assert analyze(sanitize(t))["valid"], t


def test_broken_snippets_rejected_on_right_line():
    for s in build_dataset():
        if s["label"] == "valid": continue
        r = analyze(sanitize(s["text"]))
        assert not r["valid"], s["id"]
        assert r["error"]["line"] == s["error_line"], (s["id"], r["error"])


def test_pda():
    assert pda_trace("f([1)")["problem"] and "does not match" in pda_trace("f([1)")["problem"]
    assert pda_trace("f(1))")["problem"] and pda_trace("f(1)")["ok"]


def test_sanitizer_strips_control_tokens():
    assert "<mask>" not in sanitize("x = 1 <mask>")


def test_hybrid_pipeline_with_fake_scorer():
    sc, n = FakeScorer(), 0
    for s in build_dataset():
        if s["label"] == "valid": continue
        t = sanitize(s["text"]); r2 = neural_analyze(t, sc)
        hy = hybrid_repair(r2["candidates"], analyze)
        assert hy and analyze(hy["fixed_text"])["valid"], s["id"]; n += 1
    assert n > 30


def test_semantic_names():
    sem = lambda c: analyze(c)["semantic"]["issues"]
    assert [i["name"] for i in sem("print(hello)\n")] == ["hello"]
    assert not sem('print("hello")\n') and not sem("x = 1\nprint(x)\n")
    assert [i["name"] for i in sem("def f(a):\n    return a\nprint(a)\n")] == ["a"]
    for t in load_base(): assert not sem(sanitize(t)), t


def test_semantic_repair_quotes():
    from Backend.Engine2_Probabilistic_Neural_Analyzer.semantic_optimizer import semantic_repair
    t = "print(hello)\n"; r = analyze(t)
    cands, fix = semantic_repair(t, r["semantic"], analyze)
    assert fix and fix["fixed_text"] == 'print("hello")\n', cands


def test_type_checker():
    ty = lambda c: [e["msg"] for e in analyze(c)["types"]["errors"]]
    assert ty('print("a" + 1)\n') and ty('x = 1\nx = x + "1"\n') and ty("print(len(5))\n") and ty('print(-"a")\n')
    assert ty('x = 5\nx(1)\n') and ty('print("a" - "b")\n') and ty('print(1 < "a")\n')
    for ok in ['print("a" + "b")\n', 'print("ab" * 3)\n', "print(1 + 2.5)\n", 'x = 1\nx = "a"\nprint(x + 1)\n',
               "def f(a):\n    return a + 1\nprint(f(2))\n", 'print("%d" % 5)\n']:
        assert not ty(ok), ok
    for t in load_base(): assert not ty(sanitize(t)), t


def test_type_repair():
    from Backend.Engine2_Probabilistic_Neural_Analyzer.semantic_optimizer import type_repair
    t = 'print("age: " + 5)\n'
    _, fix = type_repair(t, analyze(t)["types"], analyze)
    assert fix and fix["fixed_text"] == 'print("age: " + str(5))\n'


def test_executor():
    from Backend.executor import run_code
    assert run_code("print(1 + 2)\n")["stdout"] == "3\n"
    e = run_code('print("a" + 1)\n')["error"]; assert e["type"] == "TypeError" and e["line"] == 1
    assert run_code("print(open)\n")["error"]["type"] == "NameError"
    assert run_code("print(1 / 0)\n")["error"]["type"] == "ZeroDivisionError"
    assert run_code("while 1 < 2:\n    pass\n", timeout=1)["timed_out"]


def test_generator_and_localizer_plumbing():
    from training.generator import make_split, deep_dataset
    from training.evaluation_utils import evaluate, parser_predict
    tr = make_split(60, 1); te = make_split(40, 2, exclude={x["text"] for x in tr})
    assert not {x["text"] for x in tr} & {x["text"] for x in te}
    assert evaluate(parser_predict, te)["accuracy"] == 1.0
    deep = deep_dataset([6, 12], 3); assert evaluate(parser_predict, deep)["accuracy"] == 1.0
    from Backend.Engine2_Probabilistic_Neural_Analyzer.finetuned_detector import line_means
    assert line_means([0, 1, 5], [0.0, 1.0, 0.5], "ab\ncd") == {1: 0.5, 2: 0.5}
    # a localizer plugs into neural_analyze: it picks the line, the (fake) MLM proposes the fix
    t = "if 1 < 2\n    print(3)\n"
    r = neural_analyze(t, FakeScorer(), localizer=lambda text: (True, 1, 0.9))
    assert r["detected"] and r["predicted_line"] == 1 and hybrid_repair(r["candidates"], analyze)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("pass", k)
