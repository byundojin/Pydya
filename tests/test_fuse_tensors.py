"""Tensor 표현식 융합 패스 정합성 테스트.

C 확장 빌드가 필요한 일부 테스트는 빌드 안 됐을 때 skip.
"""

import math
import textwrap

import pytest

from pydya import compile_source

Tensor_mod = pytest.importorskip("pydya._tensor")
Tensor = Tensor_mod.Tensor
madd = Tensor_mod.madd


# ─── AST 패스: 패턴 감지 ────────────────────────────────────────────────


def _compile(src):
    return compile_source(textwrap.dedent(src))


def test_fma_pattern_with_annotations():
    out = _compile(
        """
        def f(a: Tensor, b: Tensor, c: Tensor):
            return a * b + c
        """
    )
    assert "__pydya_t.madd(a, b, c)" in out
    assert "import pydya._tensor as __pydya_t" in out


def test_commutative_add_form():
    out = _compile(
        """
        def f(a: Tensor, b: Tensor, c: Tensor):
            return c + a * b
        """
    )
    assert "__pydya_t.madd(a, b, c)" in out


def test_string_annotation_also_recognized():
    out = _compile(
        """
        def f(a: 'Tensor', b: 'Tensor', c: 'Tensor'):
            return a * b + c
        """
    )
    assert "__pydya_t.madd" in out


def test_no_annotation_no_fusion():
    out = _compile(
        """
        def f(a, b, c):
            return a * b + c
        """
    )
    assert "madd" not in out
    assert "a * b + c" in out
    assert "__pydya_t" not in out  # import 도 안 들어감


def test_scalar_operand_blocks_fusion():
    out = _compile(
        """
        def f(a: Tensor, b: Tensor):
            return a * b + 1.0
        """
    )
    assert "madd" not in out


def test_partial_annotation_does_not_fuse():
    # c 가 어노테이션 안 됐으면 c 가 Tensor 인지 컴파일러는 모름 → 융합 X.
    out = _compile(
        """
        def f(a: Tensor, b: Tensor, c):
            return a * b + c
        """
    )
    assert "madd" not in out


def test_subscript_operands_not_fused():
    out = _compile(
        """
        def f(a: Tensor, b: Tensor, c: Tensor):
            return a[0] * b[0] + c[0]
        """
    )
    assert "madd" not in out


def test_unrelated_binop_left_alone():
    out = _compile(
        """
        def f(a: Tensor, b: Tensor):
            return a - b
        """
    )
    assert "madd" not in out
    assert "import pydya._tensor" not in out


def test_module_level_expression_not_fused():
    # 함수 밖이라 어노테이션 컨텍스트가 없음.
    out = _compile(
        """
        def make_a(): return None
        def make_b(): return None
        def make_c(): return None
        a = make_a()
        b = make_b()
        c = make_c()
        d = a * b + c
        """
    )
    assert "madd" not in out


# ─── End-to-end: 컴파일된 코드가 실제로 동작하는가 ──────────────────────


def _exec_compiled(src):
    compiled = compile_source(textwrap.dedent(src))
    ns = {}
    exec(compiled, ns)
    return ns, compiled


def test_fused_function_runs_and_matches_unfused():
    src = """
        def fma(a: Tensor, b: Tensor, c: Tensor):
            return a * b + c
    """
    ns, compiled = _exec_compiled(src)
    assert "madd" in compiled
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0])
    c = Tensor([100.0, 200.0, 300.0, 400.0])
    result = ns["fma"](a, b, c)
    expected = [
        1.0 * 10.0 + 100.0,
        2.0 * 20.0 + 200.0,
        3.0 * 30.0 + 300.0,
        4.0 * 40.0 + 400.0,
    ]
    for a_val, b_val in zip(result.to_list(), expected):
        assert math.isclose(a_val, b_val, rel_tol=1e-6)


def test_madd_size_mismatch_raises():
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0, 2.0])
    c = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        madd(a, b, c)


def test_madd_wrong_types_raises():
    a = Tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        madd(a, a, [1.0, 2.0])  # 3번째가 Tensor 아님
