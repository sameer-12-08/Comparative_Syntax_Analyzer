"""Benchmark data: valid MiniPy snippets + programmatically injected single-edit syntax errors."""
import ast, random, re

BASE_TEXT = '''
def add(a, b):
    return a + b

print(add(2, 3))
---
x = 10
if x > 5:
    print("big")
else:
    print("small")
---
count = 0
while count < 3:
    count = count + 1
    print(count)
---
values = [1, 2, 3]
print(len(values))
---
def is_even(n):
    if n % 2 == 0:
        return True
    return False
---
name = "sam"
greeting = "hello " + name
print(greeting)
---
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)

print(fact(5))
---
total = 0
i = 0
while i < 10:
    if i % 2 == 0:
        total = total + i
    i = i + 1
print(total)
---
def greet(name):
    print("hi " + name)

greet("ana")
greet("raj")
---
a = 3
b = 4
if a > b:
    print(max(a, b))
elif a == b:
    print("same")
else:
    print(min(a, b))
---
x = (1 + 2) * (3 - 1)
print(x)
---
def f(a, b):
    return (a + b) * 2

result = f(1, f(2, 3))
print(result)
---
msg = "done"
if not msg == "":
    print(msg)
---
n = 5
while n > 0:
    print(n)
    n = n - 1
---
def area(w, h):
    return w * h

sizes = [area(2, 3), area(4, 5)]
print(sizes)
'''
TYPES = ["missing_colon", "missing_close_paren", "extra_close_paren", "unterminated_string"]


def _lines_with(t, pred):
    return [i for i, l in enumerate(t.split("\n")) if pred(l)]


def missing_colon(t, r):
    idx = _lines_with(t, lambda l: l.rstrip().endswith(":"))
    if not idx: return None
    i = r.choice(idx); ls = t.split("\n"); ls[i] = ls[i].rstrip()[:-1]
    return "\n".join(ls), i + 1


def missing_close_paren(t, r):
    pos = [m.start() for m in re.finditer(r"\)", t)]
    if not pos: return None
    p = r.choice(pos); return t[:p] + t[p + 1:], t.count("\n", 0, p) + 1


def extra_close_paren(t, r):
    pos = [m.start() for m in re.finditer(r"\)", t)]
    if not pos: return None
    p = r.choice(pos); return t[:p + 1] + ")" + t[p + 1:], t.count("\n", 0, p) + 1


def unterminated_string(t, r):
    ms = list(re.finditer(r"\"[^\"\n]*\"", t))
    if not ms: return None
    m = r.choice(ms); e = m.end() - 1; return t[:e] + t[e + 1:], t.count("\n", 0, e) + 1


INJECTORS = dict(zip(TYPES, [missing_colon, missing_close_paren, extra_close_paren, unterminated_string]))


def load_base():
    return [s.strip("\n") for s in BASE_TEXT.strip("\n").split("\n---\n")]


def build_dataset(seed=7):
    r, data = random.Random(seed), []
    for n, base in enumerate(load_base()):
        data.append(dict(id=f"s{n:02d}_valid", label="valid", text=base, original=base, error_line=None))
        for typ, inj in INJECTORS.items():
            res = inj(base, r)
            if not res: continue
            broken, line = res
            try:
                ast.parse(broken); continue  # keep only edits CPython also rejects
            except SyntaxError:
                pass
            data.append(dict(id=f"s{n:02d}_{typ}", label=typ, text=broken, original=base, error_line=line))
    return data
