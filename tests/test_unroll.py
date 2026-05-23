import textwrap

import pytest

from pydya import compile_source
from pydya.passes.unroll import DEFAULT_THRESHOLD, UnrollError


def _compile(src, env=None):
    return compile_source(textwrap.dedent(src), env=env or {}).strip()


# ─── attr opt-in 정책: 명시 마커가 있는 for 만 펼침 ───────────────────────────


def test_dot_product_unroll_matches_expected():
    # 당신 예시 그대로. attr[{'unroll': True}] 한 줄을 붙이면 W=4 로 펼침.
    src = """
        from pydya import attr
        W = CompileVar('W')

        def dot_product(a, b):
            result = 0
            attr[{'unroll': True}]
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


def test_marker_required_no_unroll_without_it():
    # CompileVar 로 range 가 정해져도 attr 없으면 펼치지 않는다.
    out = _compile(
        "W = CompileVar('W')\n"
        "for i in range(W):\n"
        "    print(i)\n",
        {"W": 3},
    )
    assert "for i in range(3):" in out


def test_literal_range_with_marker_unrolls():
    # 리터럴 상수여도 명시 marker 가 있으면 펼친다(사용자 의지).
    out = _compile(
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(4):\n"
        "    print(i)\n"
    )
    assert "for i" not in out
    for value in ("print(0)", "print(1)", "print(2)", "print(3)"):
        assert value in out


def test_compile_var_with_marker_unrolls():
    out = _compile(
        "from pydya import attr\n"
        "W = CompileVar('W')\n"
        "attr[{'unroll': True}]\n"
        "for i in range(W):\n"
        "    print(i * 2)\n",
        {"W": 3},
    )
    assert "for i" not in out
    for value in ("print(0)", "print(2)", "print(4)"):
        assert value in out


def test_start_stop_step_unrolls():
    out = _compile(
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(2, 8, 2):\n"
        "    print(i)\n"
    )
    assert "for i" not in out
    assert "print(2)" in out and "print(4)" in out and "print(6)" in out
    assert "print(8)" not in out  # stop 은 제외


def test_above_threshold_raises():
    n = DEFAULT_THRESHOLD + 1
    src = (
        "from pydya import attr\n"
        f"attr[{{'unroll': True}}]\n"
        f"for i in range({n}):\n"
        "    print(i)\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_nonconstant_range_raises():
    # marker 가 있는데 range 인자가 정적 상수로 좁혀지지 않으면 명시적 에러.
    src = (
        "from pydya import attr\n"
        "def f(n):\n"
        "    attr[{'unroll': True}]\n"
        "    for i in range(n):\n"
        "        print(i)\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_break_in_body_raises():
    src = (
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(3):\n"
        "    if i == 1:\n"
        "        break\n"
        "    print(i)\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_continue_in_body_raises():
    src = (
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(3):\n"
        "    if i == 1:\n"
        "        continue\n"
        "    print(i)\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_for_else_raises():
    src = (
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(3):\n"
        "    print(i)\n"
        "else:\n"
        "    print('done')\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_loop_var_store_in_body_raises():
    src = (
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(3):\n"
        "    i = 99\n"
        "    print(i)\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)


def test_nested_markers_both_unroll():
    out = _compile(
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(2):\n"
        "    attr[{'unroll': True}]\n"
        "    for j in range(2):\n"
        "        print(i, j)\n"
    )
    assert "for i" not in out and "for j" not in out
    for pair in ("print(0, 0)", "print(0, 1)", "print(1, 0)", "print(1, 1)"):
        assert pair in out


def test_break_inside_nested_loop_does_not_block_outer_unroll():
    # 안쪽 while 의 break 는 바깥 for 와 무관.
    out = _compile(
        "from pydya import attr\n"
        "attr[{'unroll': True}]\n"
        "for i in range(2):\n"
        "    while True:\n"
        "        break\n"
        "    print(i)\n"
    )
    assert "for i in range(2):" not in out
    assert "print(0)" in out and "print(1)" in out


def test_empty_range_eliminates_loop():
    out = _compile(
        "from pydya import attr\n"
        "W = CompileVar('W')\n"
        "attr[{'unroll': True}]\n"
        "for i in range(W):\n"
        "    print(i)\n",
        {"W": 0},
    )
    assert "range" not in out
    assert "print" not in out


def test_parallel_and_unroll_conflict():
    src = (
        "from pydya import attr\n"
        "out = [0] * 3\n"
        "attr[{'parallel': True, 'unroll': True}]\n"
        "for i in range(3):\n"
        "    out[i] = i\n"
    )
    with pytest.raises(UnrollError):
        compile_source(src)
