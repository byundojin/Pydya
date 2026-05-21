import pytest

from pydya import CompileVar, specialize
from pydya.passes.collect import MissingCompileValue


def test_folds_compile_var_local():
    @specialize({"V": 3})
    def f(a):
        V = CompileVar[int]()
        return a + V

    assert f(10) == 13
    assert "CompileVar" not in f.__pydya_source__


def test_prunes_static_branch():
    @specialize({"flag": True})
    def g(a):
        flag = CompileVar()
        if flag:
            return a + 1
        else:
            return a - 1

    assert g(10) == 11
    assert "a - 1" not in g.__pydya_source__


def test_string_label_form():
    @specialize({"scale": 5})
    def h(a):
        s = CompileVar("scale")
        return a * s

    assert h(4) == 20


def test_missing_value_raises():
    with pytest.raises(MissingCompileValue):

        @specialize({})
        def bad():
            V = CompileVar[int]()
            return V


def test_preserves_name_and_signature():
    @specialize({"V": 1})
    def named(a, b):
        V = CompileVar[int]()
        return a + b + V

    assert named.__name__ == "named"
    assert named(2, 3) == 6
