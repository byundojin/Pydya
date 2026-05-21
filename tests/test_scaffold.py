import ast

from pydya import CompileVar, compile_source


def test_compilevar_repr():
    assert repr(CompileVar("V")) == "CompileVar('V')"


def test_passthrough_is_valid_python():
    src = "x = 1 + 2\nprint(x)\n"
    out = compile_source(src)
    # No passes yet: output must still be semantically parseable Python.
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))


def test_env_accepts_mapping():
    assert compile_source("a = 1\n", env={"V": 3}).strip() == "a = 1"
