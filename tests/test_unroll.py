import textwrap

from pydya import compile_source
from pydya.passes.unroll import DEFAULT_THRESHOLD


def _compile(src, env=None):
    return compile_source(textwrap.dedent(src), env=env or {}).strip()


def test_dot_product_unroll_matches_expected():
    src = """
        W = CompileVar('W')

        def dot_product(a, b):
            result = 0
            for i in range(W):
                result += a[i] * b[i]
            return result
        """
    expected = textwrap.dedent(
        """
        def dot_product(a, b):
            result = 0
            result += a[0] * b[0]
            result += a[1] * b[1]
            result += a[2] * b[2]
            result += a[3] * b[3]
            return result
        """
    ).strip()
    assert _compile(src, {"W": 4}) == expected


def test_range_with_start_stop_step():
    src = """
        for i in range(2, 8, 2):
            print(i)
        """
    out = _compile(src)
    assert "for i in range" not in out
    assert "print(2)" in out and "print(4)" in out and "print(6)" in out
    assert "print(8)" not in out  # stop 은 제외


def test_count_above_threshold_not_unrolled():
    n = DEFAULT_THRESHOLD + 1
    out = _compile(f"for i in range({n}):\n    print(i)\n")
    assert f"for i in range({n}):" in out


def test_nonconstant_range_left_alone():
    out = _compile("def f(n):\n    for i in range(n):\n        print(i)\n")
    assert "for i in range(n):" in out


def test_break_in_body_skips_unroll():
    out = _compile(
        "for i in range(3):\n"
        "    if i == 1:\n"
        "        break\n"
        "    print(i)\n"
    )
    assert "for i in range(3):" in out


def test_continue_in_body_skips_unroll():
    out = _compile(
        "for i in range(3):\n"
        "    if i == 1:\n"
        "        continue\n"
        "    print(i)\n"
    )
    assert "for i in range(3):" in out


def test_for_else_skips_unroll():
    out = _compile(
        "for i in range(3):\n"
        "    print(i)\n"
        "else:\n"
        "    print('done')\n"
    )
    assert "for i in range(3):" in out


def test_loop_var_store_in_body_skips_unroll():
    out = _compile(
        "for i in range(3):\n"
        "    i = 99\n"
        "    print(i)\n"
    )
    assert "for i in range(3):" in out


def test_nested_static_for_both_unrolled():
    out = _compile(
        "for i in range(2):\n"
        "    for j in range(2):\n"
        "        print(i, j)\n"
    )
    assert "for i" not in out and "for j" not in out
    for pair in ("print(0, 0)", "print(0, 1)", "print(1, 0)", "print(1, 1)"):
        assert pair in out


def test_break_inside_nested_loop_does_not_block_outer_unroll():
    # 안쪽 while 의 break 는 우리 for 와 무관하다.
    out = _compile(
        "for i in range(2):\n"
        "    while True:\n"
        "        break\n"
        "    print(i)\n"
    )
    assert "for i in range(2):" not in out
    assert "print(0)" in out and "print(1)" in out


def test_empty_range_eliminates_loop():
    out = _compile("for i in range(0):\n    print(i)\n")
    assert "range" not in out
    assert "print" not in out


def test_compile_var_in_range_unrolls():
    out = _compile(
        "W = CompileVar('W')\n"
        "for i in range(W):\n"
        "    print(i * 2)\n",
        {"W": 3},
    )
    assert "for i" not in out
    for value in ("print(0)", "print(2)", "print(4)"):
        assert value in out
