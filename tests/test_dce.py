import ast

from pydya.passes.dce import eliminate_dead_code


def _run(src):
    tree = ast.parse(src)
    eliminate_dead_code(tree)
    return ast.unparse(tree).strip()


def test_removes_unused_pure_assignment_in_function():
    src = "def f():\n    dead = 1 + 2\n    x = 5\n    return x\n"
    assert _run(src) == "def f():\n    x = 5\n    return x"


def test_keeps_used_assignment_in_function():
    src = "def f():\n    a = 8\n    return a\n"
    assert _run(src) == "def f():\n    a = 8\n    return a"


def test_keeps_side_effecting_rhs():
    src = "def f():\n    a = compute()\n    return 0\n"
    assert _run(src) == "def f():\n    a = compute()\n    return 0"


def test_fixpoint_removes_chained_dead_stores():
    src = "def f():\n    a = 1\n    b = a + 1\n    c = 9\n    return c\n"
    assert _run(src) == "def f():\n    c = 9\n    return c"


def test_module_top_level_bindings_are_preserved():
    # 최상위 이름은 export 일 수 있으므로 절대 제거하지 않는다.
    assert _run("a = 1\n") == "a = 1"
