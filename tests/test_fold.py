import ast

from pydya.passes.fold import fold


def _fold(src, static):
    tree = ast.parse(src)
    fold(tree, static)
    return ast.unparse(tree).strip()


def test_substitutes_static_name():
    assert _fold("x = V\n", {"V": 3}) == "x = 3"


def test_folds_static_into_runtime_expression():
    assert _fold("b = a + V\n", {"V": 3}) == "b = a + 3"


def test_folds_pure_constant_comparison():
    assert _fold("c = V < 5\n", {"V": 3}) == "c = True"


def test_folds_nested_arithmetic():
    assert _fold("x = (V + 2) * 10\n", {"V": 3}) == "x = 50"


def test_does_not_propagate_runtime_binding():
    # 'a' is a runtime binding, so its use is not substituted even though
    # the right-hand side is constant.
    assert _fold("a = 8\nb = a + V\n", {"V": 3}) == "a = 8\nb = a + 3"


def test_boolop_short_circuit():
    assert _fold("x = V > 0 and V < 10\n", {"V": 3}) == "x = True"


def test_unary():
    assert _fold("x = -V\n", {"V": 3}) == "x = -3"
