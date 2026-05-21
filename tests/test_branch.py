import ast

from pydya.passes.branch import eliminate_branches
from pydya.passes.fold import fold


def _run(src, static):
    tree = ast.parse(src)
    fold(tree, static)
    eliminate_branches(tree)
    return ast.unparse(tree).strip()


def test_keeps_taken_branch():
    src = "if V < 5:\n    a = 1\nelse:\n    a = 2\n"
    assert _run(src, {"V": 3}) == "a = 1"


def test_keeps_else_branch():
    src = "if V < 5:\n    a = 1\nelse:\n    a = 2\n"
    assert _run(src, {"V": 9}) == "a = 2"


def test_removes_if_without_else_when_false():
    src = "if V < 5:\n    a = 1\nb = 2\n"
    assert _run(src, {"V": 9}) == "b = 2"


def test_ifexp_folds_to_branch():
    assert _run("x = 1 if V < 5 else 2\n", {"V": 3}) == "x = 1"


def test_while_false_removed():
    src = "while V < 0:\n    a = 1\nb = 2\n"
    assert _run(src, {"V": 3}) == "b = 2"


def test_runtime_condition_untouched():
    src = "if a < 5:\n    x = 1\nelse:\n    x = 2\n"
    out = _run(src, {"V": 3})
    assert "if a < 5:" in out
