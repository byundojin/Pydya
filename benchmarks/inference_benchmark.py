"""미융합 vs 융합 추론 단계 성능 비교.

같은 계산을 두 가지로:
  - UNFUSED : ``relu(matmul(W, x) + b)`` — 임시 텐서 2개(matmul 결과, add 결과)
  - FUSED   : ``linear_relu(W, x, b)`` — 단일 할당 + 단일 패스 (matmul+add+relu
              를 한 inner loop 안에 융합. bias 는 row 별 acc 의 초기값으로 처리)

신경망 추론에서 가장 흔한 패턴(Dense + ReLU)을 직접 측정한다.
실행:  PYTHONPATH=. python benchmarks/inference_benchmark.py
"""

import os
import sys
import time

from pydya import Tensor
from pydya._tensor import linear_relu, matmul, relu

ROWS = 1024
COLS = 1024
ITERS = 200


def make_weight(rows, cols):
    data = [[((i * cols + j) % 97) / 97.0 - 0.5 for j in range(cols)] for i in range(rows)]
    return Tensor(data)


def make_vec(n, seed=0):
    return Tensor([((i + seed) % 89) / 89.0 - 0.5 for i in range(n)])


W = make_weight(ROWS, COLS)
x = make_vec(COLS, seed=1)
b = make_vec(ROWS, seed=2)


def unfused():
    return relu(matmul(W, x) + b)


def fused():
    return linear_relu(W, x, b)


def bench(fn):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
    return time.perf_counter() - start, out


def main():
    print("=" * 64)
    print(" Dense + ReLU — UNFUSED vs FUSED")
    print("=" * 64)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print(f"  W shape: ({ROWS}, {COLS}),  x: ({COLS},),  b: ({ROWS},)")
    print(f"  iters  : {ITERS}")
    print("-" * 64)

    t_unfused, r_unfused = bench(unfused)
    t_fused, r_fused = bench(fused)

    diff = max(abs(u - f) for u, f in zip(r_unfused.to_list(), r_fused.to_list()))
    print(f"  UNFUSED  relu(matmul(W, x) + b) : {t_unfused:.3f}s")
    print(f"  FUSED    linear_relu(W, x, b)   : {t_fused:.3f}s")
    print(f"  speedup                          : {t_unfused / t_fused:.2f}x")
    print(f"  max abs diff                     : {diff:g}")
    print("=" * 64)


if __name__ == "__main__":
    main()
