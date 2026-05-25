"""C 레벨 Tensor primitive 의 정합성 테스트.

C 확장 빌드가 필요하다(``python setup.py build_ext --inplace``). 빌드되지
않은 환경에서는 모듈 import 가 실패하므로 전체 테스트가 skip 된다.
"""

import math

import pytest

Tensor = pytest.importorskip("pydya._tensor").Tensor


# ─── 생성 / 길이 / 인덱싱 ────────────────────────────────────────────────


def test_construct_from_size_default_zero():
    t = Tensor(5)
    assert len(t) == 5
    assert t.to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_construct_from_size_with_fill():
    t = Tensor(4, fill=2.5)
    assert t.to_list() == [2.5, 2.5, 2.5, 2.5]


def test_construct_from_list():
    t = Tensor([1.0, 2.0, 3.0])
    assert len(t) == 3
    assert t.to_list() == [1.0, 2.0, 3.0]


def test_construct_from_tuple():
    t = Tensor((1.0, 2.0, 3.0))
    assert t.to_list() == [1.0, 2.0, 3.0]


def test_construct_negative_size_raises():
    with pytest.raises(ValueError):
        Tensor(-1)


def test_getitem_and_setitem():
    t = Tensor([1.0, 2.0, 3.0])
    assert t[0] == 1.0
    assert t[2] == 3.0
    assert t[-1] == 3.0  # 음수 인덱스
    t[1] = 99.0
    assert t.to_list() == [1.0, 99.0, 3.0]


def test_index_out_of_range_raises():
    t = Tensor([1.0, 2.0])
    with pytest.raises(IndexError):
        _ = t[5]
    with pytest.raises(IndexError):
        t[5] = 0.0


# ─── 산술: Tensor × Tensor ─────────────────────────────────────────────


def test_elementwise_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    assert (a + b).to_list() == [11.0, 22.0, 33.0]


def test_elementwise_sub():
    a = Tensor([10.0, 20.0, 30.0])
    b = Tensor([1.0, 2.0, 3.0])
    assert (a - b).to_list() == [9.0, 18.0, 27.0]


def test_elementwise_mul():
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0])
    assert (a * b).to_list() == [10.0, 40.0, 90.0, 160.0]


def test_size_mismatch_raises():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = a + b


# ─── 산술: Tensor × Scalar (양쪽) ──────────────────────────────────────


def test_tensor_times_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    assert (a * 2).to_list() == [2.0, 4.0, 6.0]
    assert (a * 2.5).to_list() == [2.5, 5.0, 7.5]


def test_scalar_times_tensor():
    a = Tensor([1.0, 2.0, 3.0])
    assert (2 * a).to_list() == [2.0, 4.0, 6.0]


def test_tensor_plus_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    assert (a + 10).to_list() == [11.0, 12.0, 13.0]
    assert (10 + a).to_list() == [11.0, 12.0, 13.0]


def test_scalar_minus_tensor_non_commutative():
    a = Tensor([1.0, 2.0, 3.0])
    assert (a - 1).to_list() == [0.0, 1.0, 2.0]
    assert (10 - a).to_list() == [9.0, 8.0, 7.0]


def test_returns_new_tensor_not_alias():
    a = Tensor([1.0, 2.0, 3.0])
    b = a * 2
    a[0] = 99.0
    # b 는 독립된 결과 — a 수정 후에도 그대로
    assert b.to_list() == [2.0, 4.0, 6.0]


# ─── 정확성 (float32 정밀도 한도 안에서) ──────────────────────────────


def test_large_size_correctness():
    n = 10_000
    a = Tensor([float(i) for i in range(n)])
    b = Tensor([float(i) * 2.0 for i in range(n)])
    result = (a + b).to_list()
    for i, v in enumerate(result):
        assert math.isclose(v, float(i) * 3.0, rel_tol=1e-6)


# ─── repr 가 죽지 않는지 ────────────────────────────────────────────────


def test_repr_small():
    assert "Tensor(" in repr(Tensor([1.0, 2.0]))


def test_repr_large_has_preview():
    r = repr(Tensor(100, fill=1.0))
    assert "size=100" in r
    assert "..." in r


# ─── N-D 생성 / shape / 인덱싱 ──────────────────────────────────────────


def test_construct_from_shape_tuple_2d():
    t = Tensor((2, 3), fill=1.5)
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.size == 6
    assert t.to_list() == [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]


def test_construct_from_nested_list_2d():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_construct_from_3d_nested_list():
    t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    assert t.shape == (2, 2, 2)
    assert t.to_list() == [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]


def test_tuple_indexing_2d():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t[0, 0] == 1.0
    assert t[1, 2] == 6.0
    assert t[-1, -1] == 6.0
    t[0, 1] = 99.0
    assert t.to_list() == [[1.0, 99.0, 3.0], [4.0, 5.0, 6.0]]


def test_int_index_on_2d_rejected():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        _ = t[0]
    with pytest.raises(TypeError):
        t[0] = 1.0


def test_tuple_index_wrong_arity_rejected():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        _ = t[0, 0, 0]


def test_shape_mismatch_in_binop_rejected():
    a = Tensor([1.0, 2.0, 3.0])         # (3,)
    b = Tensor([[1.0, 2.0, 3.0]])        # (1, 3)
    with pytest.raises(ValueError):
        _ = a + b


def test_same_shape_2d_binop():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0], [30.0, 40.0]])
    assert (a + b).to_list() == [[11.0, 22.0], [33.0, 44.0]]
    assert (a * 2).to_list() == [[2.0, 4.0], [6.0, 8.0]]


def test_floats_tuple_still_treated_as_data():
    # back-compat: 부동소수점만 들어간 튜플은 데이터로
    t = Tensor((1.0, 2.0, 3.0))
    assert t.shape == (3,)
    assert t.to_list() == [1.0, 2.0, 3.0]


# ─── matmul / relu / linear_relu ────────────────────────────────────────


from pydya._tensor import matmul, relu, linear_relu


def test_matmul_2d_1d():
    W = Tensor([[1.0, 2.0], [3.0, 4.0]])
    x = Tensor([5.0, 6.0])
    out = matmul(W, x)
    assert out.shape == (2,)
    assert out.to_list() == [17.0, 39.0]


def test_matmul_shape_mismatch_raises():
    W = Tensor([[1.0, 2.0, 3.0]])
    x = Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        matmul(W, x)


def test_matmul_wrong_ndim_raises():
    W = Tensor([1.0, 2.0])  # 1D
    x = Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        matmul(W, x)


def test_relu_elementwise():
    t = Tensor([-1.0, 2.0, -3.0, 4.0, 0.0])
    assert relu(t).to_list() == [0.0, 2.0, 0.0, 4.0, 0.0]


def test_relu_preserves_shape():
    t = Tensor([[-1.0, 2.0], [3.0, -4.0]])
    out = relu(t)
    assert out.shape == (2, 2)
    assert out.to_list() == [[0.0, 2.0], [3.0, 0.0]]


def test_linear_relu_matches_unfused():
    W = Tensor([[1.0, -2.0], [3.0, 4.0]])
    x = Tensor([5.0, 6.0])
    b = Tensor([-100.0, 10.0])
    fused = linear_relu(W, x, b).to_list()
    unfused = relu(matmul(W, x) + b).to_list()
    assert fused == unfused


def test_linear_relu_shape_check():
    W = Tensor([[1.0, 2.0]])         # (1, 2)
    x = Tensor([1.0, 2.0, 3.0])      # (3,) — cols mismatch
    b = Tensor([0.0])
    with pytest.raises(ValueError):
        linear_relu(W, x, b)


# ─── XOR MLP — 진리표 4개 케이스 ────────────────────────────────────────


def _xor_forward(x):
    """relu 기반 2→2→1 XOR MLP. 가중치 하드코딩 (학습 아님, 추론만)."""
    W1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = Tensor([0.0, -1.0])
    W2 = Tensor([[1.0, -2.0]])
    b2 = Tensor([0.0])
    h = linear_relu(W1, x, b1)
    y = linear_relu(W2, h, b2)
    return y.to_list()[0]


@pytest.mark.parametrize(
    "x_vals, expected",
    [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ],
)
def test_xor_truth_table(x_vals, expected):
    x = Tensor(x_vals)
    assert _xor_forward(x) == expected
