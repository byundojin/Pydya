import ast

import pytest

from pydya.passes.collect import MissingCompileValue, collect_static_env


def _parse(src):
    return ast.parse(src)


def test_collects_value_and_strips_declaration():
    tree = _parse("V = CompileVar('V')\na = 1\n")
    static = collect_static_env(tree, {"V": 3})
    assert static == {"V": 3}
    assert ast.unparse(tree).strip() == "a = 1"


def test_label_differs_from_variable_name():
    tree = _parse("v = CompileVar('flag')\n")
    static = collect_static_env(tree, {"flag": True})
    assert static == {"v": True}


def test_strips_pydya_import():
    tree = _parse("from pydya import CompileVar\nV = CompileVar('V')\nx = V\n")
    static = collect_static_env(tree, {"V": 5})
    assert static == {"V": 5}
    assert ast.unparse(tree).strip() == "x = V"


def test_missing_value_raises():
    tree = _parse("V = CompileVar('V')\n")
    with pytest.raises(MissingCompileValue):
        collect_static_env(tree, {})


def test_ignores_unrelated_assignments():
    tree = _parse("a = foo('V')\n")
    static = collect_static_env(tree, {})
    assert static == {}
    assert ast.unparse(tree).strip() == "a = foo('V')"


def test_typed_form_label_from_target():
    tree = _parse("V = CompileVar[int]()\nx = V\n")
    static = collect_static_env(tree, {"V": 3})
    assert static == {"V": 3}
    assert ast.unparse(tree).strip() == "x = V"


def test_no_arg_form_label_from_target():
    tree = _parse("V = CompileVar()\n")
    static = collect_static_env(tree, {"V": 7})
    assert static == {"V": 7}


def test_typed_form_missing_value_raises():
    tree = _parse("V = CompileVar[int]()\n")
    with pytest.raises(MissingCompileValue):
        collect_static_env(tree, {})


def test_attribute_typed_form():
    tree = _parse("V = pydya.CompileVar[str]()\n")
    static = collect_static_env(tree, {"V": "hi"})
    assert static == {"V": "hi"}
