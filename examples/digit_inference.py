"""손글씨 숫자 추론 — Pydya 컴파일러로 돌리는 사전학습 MLP.

학습은 외부(``tools/train_digits.py``)에서 끝났고 가중치는 JSON 으로
박혀 있다. 여기선 *추론만* — pydya Tensor 로 가중치/입력을 올리고,
``compile_source`` 가 ``relu(W @ x + b)`` 패턴을 ``linear_relu(W, x, b)``
로 lowering 한 함수로 forward pass 를 돈다.

실행:
    python setup.py build_ext --inplace    # 최초 1회 (C 확장 빌드)
    PYTHONPATH=. python examples/digit_inference.py
"""

import json
from pathlib import Path

from pydya import Tensor, compile_source

WEIGHTS_PATH = Path(__file__).resolve().parent / "digit_weights.json"

# pydya 가 compile_source 로 lowering 할 forward pass.
# - hidden = relu(W1 @ x + b1)  -> linear_relu(W1, x, b1) 로 융합
# - output = W2 @ hidden + b2   (raw logits, softmax 는 argmax 가 보존해 불필요)
FORWARD_SRC = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""


def argmax(values):
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values[1:], start=1):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def load_weights_and_samples():
    payload = json.loads(WEIGHTS_PATH.read_text())
    weights = payload["weights"]
    W1 = Tensor(weights["W1"])
    b1 = Tensor(weights["b1"])
    W2 = Tensor(weights["W2"])
    b2 = Tensor(weights["b2"])
    samples = payload["samples"]
    return W1, b1, W2, b2, samples, payload["architecture"], payload["trained_test_accuracy"]


def main():
    W1, b1, W2, b2, samples, arch, trained_acc = load_weights_and_samples()

    compiled = compile_source(FORWARD_SRC)
    print("=== compiled forward ===")
    print(compiled)

    ns = {}
    exec(compiled, ns)
    forward = ns["forward"]

    correct = 0
    print(f"\n=== inference on {len(samples['x'])} held-out samples ===")
    print(f"architecture: {arch['input']} -> {arch['hidden']} -> {arch['output']} (relu hidden)")
    print(f"trained test accuracy: {trained_acc:.4f}\n")
    for i, (x_vals, y_true, y_pred_ref) in enumerate(
        zip(samples["x"], samples["y_true"], samples["y_pred_sklearn"])
    ):
        x = Tensor(x_vals)
        logits = forward(x, W1, b1, W2, b2).to_list()
        pred = argmax(logits)
        if pred == y_true:
            correct += 1
        if i < 8:
            mark = "OK " if pred == y_true else "ERR"
            ref = "==sklearn" if pred == y_pred_ref else "!=sklearn"
            print(f"  sample {i:>2}: pred={pred}  true={y_true}  {mark}  ({ref})")
    n = len(samples["x"])
    print(f"\npydya accuracy on these {n} samples: {correct}/{n} = {correct / n:.4f}")


if __name__ == "__main__":
    main()
