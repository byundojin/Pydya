import ast

from pydya.passes.fold import fold
from pydya.passes.inline import inline_calls


def _run(src, static=None):
    tree = ast.parse(src)
    fold(tree, static or {})
    inline_calls(tree)
    return ast.unparse(tree).strip()


def test_inlines_constant_call_and_keeps_def():
    src = "def f(a):\n    return a + V\nx = f(5)\n"
    out = _run(src, {"V": 3})
    assert "def f(a):\n    return a + 3" in out
    assert "x = 8" in out


def test_does_not_inline_runtime_argument():
    src = "def f(a):\n    return a + 1\nx = f(y)\n"
    out = _run(src)
    assert "x = f(y)" in out


def test_ignores_functions_with_defaults():
    src = "def f(a=1):\n    return a\nx = f(5)\n"
    out = _run(src)
    assert "x = f(5)" in out


def test_ignores_multistatement_functions():
    src = "def f(a):\n    b = a\n    return b\nx = f(5)\n"
    out = _run(src)
    assert "x = f(5)" in out
