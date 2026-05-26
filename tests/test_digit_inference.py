"""사전학습된 8x8 손글씨 숫자 MLP 추론 정합성 테스트.

사전학습 가중치(``examples/digit_weights.json``)와 50개의 held-out 샘플은
``tools/train_digits.py`` 가 생성해 repo 에 박혀 있다. 이 테스트는:

* 컴파일 결과가 ``linear_relu`` 융합을 포함하는지
* pydya 추론 정확도가 학습 시 측정한 sklearn 의 test 정확도와 거의 같은지
* 개별 예측이 학습기(sklearn)의 예측과 일치하는지 (float32 라운딩 한도 안)
"""

import json
from pathlib import Path

import pytest

Tensor_mod = pytest.importorskip("pydya._tensor")
Tensor = Tensor_mod.Tensor

from pydya import compile_source

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "examples" / "digit_weights.json"

FORWARD_SRC = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""


def _argmax(values):
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values[1:], start=1):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


@pytest.fixture(scope="module")
def assets():
    payload = json.loads(WEIGHTS_PATH.read_text())
    w = payload["weights"]
    W1 = Tensor(w["W1"])
    b1 = Tensor(w["b1"])
    W2 = Tensor(w["W2"])
    b2 = Tensor(w["b2"])
    s = payload["samples"]
    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "samples": s,
        "trained_test_accuracy": payload["trained_test_accuracy"],
    }


@pytest.fixture(scope="module")
def forward_fn():
    compiled = compile_source(FORWARD_SRC)
    assert "__pydya_t.linear_relu(W1, x, b1)" in compiled, \
        "fuse_tensors 가 relu(W @ x + b) 를 linear_relu 로 융합해야 한다"
    ns = {}
    exec(compiled, ns)
    return ns["forward"]


def test_compile_source_produces_linear_relu_fusion():
    # forward_fn fixture 가 assert 로 한 번 확인하지만, 단독 회귀 테스트로도 둔다.
    compiled = compile_source(FORWARD_SRC)
    assert "__pydya_t.linear_relu(W1, x, b1)" in compiled
    assert "import pydya._tensor as __pydya_t" in compiled


def test_pydya_predictions_match_sklearn_one_by_one(assets, forward_fn):
    """sklearn 이 같은 가중치로 만든 예측과 pydya 의 예측이 샘플별로 같다."""
    s = assets["samples"]
    n_match = 0
    for x_vals, y_pred_sklearn in zip(s["x"], s["y_pred_sklearn"]):
        x = Tensor(x_vals)
        logits = forward_fn(x, assets["W1"], assets["b1"], assets["W2"], assets["b2"]).to_list()
        if _argmax(logits) == y_pred_sklearn:
            n_match += 1
    # float32 vs float64 라운딩으로 극소수 어긋날 수 있어 95% 임계.
    n = len(s["x"])
    assert n_match / n >= 0.95, f"pydya vs sklearn 일치 {n_match}/{n}"


def test_pydya_accuracy_matches_trained_accuracy(assets, forward_fn):
    """pydya 의 정확도가 학습 시점 sklearn 의 test 정확도와 비슷하다 (±5%p)."""
    s = assets["samples"]
    correct = 0
    for x_vals, y_true in zip(s["x"], s["y_true"]):
        x = Tensor(x_vals)
        logits = forward_fn(x, assets["W1"], assets["b1"], assets["W2"], assets["b2"]).to_list()
        if _argmax(logits) == y_true:
            correct += 1
    n = len(s["x"])
    accuracy = correct / n
    trained = assets["trained_test_accuracy"]
    assert abs(accuracy - trained) <= 0.05, \
        f"pydya accuracy {accuracy:.4f} vs trained {trained:.4f}"
