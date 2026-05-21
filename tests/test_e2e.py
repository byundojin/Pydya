import ast
import io
import contextlib

from pydya import compile_source

SOURCE = """\
V = CompileVar('V')

def f(a):
    return a + V

if V < 5:
    a = f(5)
else:
    a = 5

b = a + V

print(a)
print(b)
"""


def _normalize(src):
    return ast.unparse(ast.parse(src)).strip()


def test_readme_example_compiles_to_expected_source():
    expected = """\
def f(a):
    return a + 3
a = 8
b = a + 3
print(a)
print(b)
"""
    out = compile_source(SOURCE, env={"V": 3})
    assert _normalize(out) == _normalize(expected)


def test_compiled_output_runs_with_same_observable_behaviour():
    out = compile_source(SOURCE, env={"V": 3})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(out, {})
    assert buf.getvalue() == "8\n11\n"


def test_else_branch_selected_for_large_value():
    out = compile_source(SOURCE, env={"V": 9})
    # V < 5 가 거짓이므로 'a = 5' 가 선택되고 f 는 호출되지 않는다.
    assert "a = 5" in out
    assert "f(5)" not in out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(out, {})
    assert buf.getvalue() == "5\n14\n"
