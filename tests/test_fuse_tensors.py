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


# ─── linear_relu 융합 패턴 ─────────────────────────────────────────────


def test_linear_relu_via_at_operator():
    out = _compile(
        """
        def f(W: Tensor, x: Tensor, b: Tensor):
            return relu(W @ x + b)
        """
    )
    assert "__pydya_t.linear_relu(W, x, b)" in out
    assert "import pydya._tensor as __pydya_t" in out


def test_linear_relu_via_matmul_call():
    out = _compile(
        """
        def f(W: Tensor, x: Tensor, b: Tensor):
            return relu(matmul(W, x) + b)
        """
    )
    assert "__pydya_t.linear_relu(W, x, b)" in out


def test_linear_relu_commutative_add():
    out = _compile(
        """
        def f(W: Tensor, x: Tensor, b: Tensor):
            return relu(b + W @ x)
        """
    )
    assert "__pydya_t.linear_relu(W, x, b)" in out


def test_tensor_local_propagation_through_assignment():
    # h 가 Tensor-producing 식에서 왔으므로 두 번째 패턴도 융합돼야 한다.
    out = _compile(
        """
        def f(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
            h = relu(W1 @ x + b1)
            return relu(W2 @ h + b2)
        """
    )
    assert out.count("__pydya_t.linear_relu") == 2


def test_bare_tensor_func_names_get_qualified():
    # 어노테이션 있는 함수에서 bare matmul/relu 호출이 qualify 된다
    out = _compile(
        """
        def f(W: Tensor, x: Tensor):
            return matmul(W, x)
        """
    )
    assert "__pydya_t.matmul(W, x)" in out


def test_no_annotation_no_qualify():
    # 어노테이션 없는 함수의 bare matmul 은 손대지 않는다 (Tensor 컨텍스트 없음)
    out = _compile(
        """
        def f(a, b):
            return matmul(a, b)
        """
    )
    assert "matmul(a, b)" in out
    assert "__pydya_t" not in out


def test_partial_annotation_no_fusion():
    # b 가 어노테이션 안 됐으면 융합 X
    out = _compile(
        """
        def f(W: Tensor, x: Tensor, b):
            return relu(W @ x + b)
        """
    )
    assert "linear_relu" not in out


# ─── end-to-end: 실제 실행해서 결과 확인 ────────────────────────────────


def test_linear_relu_compiled_function_runs():
    ns, compiled = _exec_compiled(
        """
        def step(W: Tensor, x: Tensor, b: Tensor):
            return relu(W @ x + b)
        """
    )
    assert "linear_relu" in compiled
    W = Tensor([[1.0, -2.0], [3.0, 4.0]])
    x = Tensor([5.0, 6.0])
    b = Tensor([-100.0, 10.0])
    out = ns["step"](W, x, b)
    # relu(W@x + b) = relu([1*5-2*6, 3*5+4*6] + [-100, 10]) = relu([-107, 49]) = [0, 49]
    assert out.to_list() == [0.0, 49.0]


def test_xor_mlp_via_compile_source():
    """XOR MLP 가 compile_source 를 통해 융합되고, 진리표 4개를 맞추는지."""
    ns, compiled = _exec_compiled(
        """
        def xor_mlp(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
            h = relu(W1 @ x + b1)
            return relu(W2 @ h + b2)
        """
    )
    assert compiled.count("__pydya_t.linear_relu") == 2

    W1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = Tensor([0.0, -1.0])
    W2 = Tensor([[1.0, -2.0]])
    b2 = Tensor([0.0])

    fwd = ns["xor_mlp"]
    for x_vals, expected in [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]:
        x = Tensor(x_vals)
        out = fwd(x, W1, b1, W2, b2)
        assert out.to_list() == [expected]
