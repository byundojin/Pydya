import ast

from pydya import CompileVar, compile_source


def test_compilevar_repr():
    assert repr(CompileVar("V")) == "CompileVar('V')"


def test_output_is_valid_python():
    src = "x = a + 1\nprint(x)\n"
    out = compile_source(src)
    ast.parse(out)  # 출력은 반드시 파싱 가능해야 한다
    assert "print(x)" in out


def test_env_accepts_mapping():
    assert compile_source("a = 1\n", env={"V": 3}).strip() == "a = 1"
