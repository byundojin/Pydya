"""오프라인 학습 스크립트 — sklearn 의 8x8 손글씨 숫자(load_digits) 로 작은
MLP 를 학습하고, 그 가중치와 일부 테스트 샘플을 JSON 으로 저장한다.

**이 스크립트 자체는 pydya 의 일부가 아니다 (sklearn/numpy 의존).**
한 번 실행해 가중치를 생성하면 ``examples/digit_inference.py`` 가 그 JSON 을
순수 pydya 로 로드해 추론한다 — *학습은 외부, 추론은 우리 컴파일러* 의 흐름.

실행 (오프라인, 한 번):  python tools/train_digits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

OUT_PATH = Path(__file__).resolve().parent.parent / "examples" / "digit_weights.json"

# 아키텍처: 64 → 32 → 10. ReLU hidden (pydya 가 가진 활성화), softmax 출력은
# argmax 가 보존하므로 추론 시점에선 raw logit 의 argmax 만 계산.
HIDDEN = 32
SEED = 0
N_TEST_SAMPLES = 50  # 추론 예시에 함께 박을 테스트 샘플 수


def main():
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # 0..16 → 0..1 정규화
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN,),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test  accuracy: {test_acc:.4f}")

    # sklearn 의 MLPClassifier 가중치: coefs_[i] shape (in, out), intercepts_[i] shape (out,)
    # pydya 의 matmul 은 W @ x 형태 (out, in) × (in,) 라서 전치 필요.
    W1 = clf.coefs_[0].T.astype(np.float32)      # (HIDDEN, 64)
    b1 = clf.intercepts_[0].astype(np.float32)   # (HIDDEN,)
    W2 = clf.coefs_[1].T.astype(np.float32)      # (10, HIDDEN)
    b2 = clf.intercepts_[1].astype(np.float32)   # (10,)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=N_TEST_SAMPLES, replace=False)
    samples = X_test[idx]
    labels = y_test[idx]
    sample_preds = clf.predict(samples)

    payload = {
        "architecture": {
            "input": int(X.shape[1]),
            "hidden": HIDDEN,
            "output": 10,
            "activation": "relu",
        },
        "weights": {
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist(),
        },
        "samples": {
            # pydya 추론 시 numpy 없이도 비교할 수 있도록 sample/label/sklearn-pred 다 박는다.
            "x": samples.tolist(),
            "y_true": labels.tolist(),
            "y_pred_sklearn": sample_preds.tolist(),
        },
        "trained_test_accuracy": float(test_acc),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
