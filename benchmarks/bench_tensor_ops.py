"""C Tensor 의 단일 연산 — Python list 대비 raw 가속.

목적: C 레벨 Tensor 의 *순수* 가속을 측정. 인터프리터 우회 + contiguous 메모리
+ alloc 1회 + auto-vectorize 의 *총합* 이 element-wise 연산에서 얼마나 나오는지.

연산: add / sub / mul / matmul.  N 을 sweep 해서 cache 효과까지 가시화.
각 측정은 분포(min/p50/p95/p99/std) 까지 본다.

실행:  PYTHONPATH=. python benchmarks/bench_tensor_ops.py
"""

from __future__ import annotations

import os
import random
import sys

from pydya import Tensor
from pydya._tensor import matmul

from _stats import Stats, fmt_time, print_header, print_row, time_samples

random.seed(0)


# ─── Python list 베이스라인 ─────────────────────────────────────────────


def py_add(a, b):
    return [x + y for x, y in zip(a, b)]


def py_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def py_mul(a, b):
    return [x * y for x, y in zip(a, b)]


def py_matvec(W, x):
    """순수 Python matrix(rows×cols) @ vector(cols) → vector(rows)."""
    rows = len(W)
    cols = len(x)
    out = [0.0] * rows
    for i in range(rows):
        row = W[i]
        acc = 0.0
        for j in range(cols):
            acc += row[j] * x[j]
        out[i] = acc
    return out


# ─── 측정 ──────────────────────────────────────────────────────────────


def bench_elementwise(name, py_fn, tensor_op_str, sizes):
    """element-wise add/sub/mul 한 op 에 대해 list vs Tensor 비교."""
    print(f"\n## {name} — element-wise {name.split()[-1].lower()}")
    print(f"{'N':>10}    {'list':>22}    {'Tensor':>22}    median_speedup")
    print("─" * 92)
    for n in sizes:
        a_py = [random.random() for _ in range(n)]
        b_py = [random.random() for _ in range(n)]
        a_t = Tensor(a_py)
        b_t = Tensor(b_py)
        # 측정 횟수는 N 에 따라 조절 — 작을수록 더 많이 inner repeat
        inner = max(1, 2000 // max(1, n // 100))
        iters = 50

        samples_py = time_samples(lambda: py_fn(a_py, b_py), iters=iters, inner=inner)
        s_py = Stats(samples_py)

        if tensor_op_str == "add":
            samples_t = time_samples(lambda: a_t + b_t, iters=iters, inner=inner)
        elif tensor_op_str == "sub":
            samples_t = time_samples(lambda: a_t - b_t, iters=iters, inner=inner)
        elif tensor_op_str == "mul":
            samples_t = time_samples(lambda: a_t * b_t, iters=iters, inner=inner)
        else:
            raise ValueError(tensor_op_str)
        s_t = Stats(samples_t)

        speedup = s_py.p50 / s_t.p50 if s_t.p50 > 0 else float("inf")
        print(
            f"{n:>10}    "
            f"{fmt_time(s_py.p50)} ±{fmt_time(s_py.std)}    "
            f"{fmt_time(s_t.p50)} ±{fmt_time(s_t.std)}    "
            f"{speedup:>7.1f}x"
        )


def bench_matmul(sizes):
    """matrix-vector matmul — rows × cols 다양 사이즈."""
    print(f"\n## matmul (2D × 1D)")
    print(f"{'shape':>16}    {'list':>22}    {'Tensor':>22}    median_speedup")
    print("─" * 92)
    for rows, cols in sizes:
        W_py = [[random.random() for _ in range(cols)] for _ in range(rows)]
        x_py = [random.random() for _ in range(cols)]
        W_t = Tensor(W_py)
        x_t = Tensor(x_py)

        # N 이 크면 inner 줄여 측정 시간 제한
        ops = rows * cols
        inner = max(1, 100_000 // ops) if ops > 0 else 1
        iters = 50

        samples_py = time_samples(lambda: py_matvec(W_py, x_py), iters=iters, inner=inner)
        s_py = Stats(samples_py)
        samples_t = time_samples(lambda: matmul(W_t, x_t), iters=iters, inner=inner)
        s_t = Stats(samples_t)
        speedup = s_py.p50 / s_t.p50 if s_t.p50 > 0 else float("inf")
        print(
            f"{rows:>5}×{cols:<5}      "
            f"{fmt_time(s_py.p50)} ±{fmt_time(s_py.std)}    "
            f"{fmt_time(s_t.p50)} ±{fmt_time(s_t.std)}    "
            f"{speedup:>7.1f}x"
        )


def main():
    print("=" * 92)
    print(" C Tensor 단일 연산 — Python list 대비 가속")
    print("=" * 92)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")

    sizes_ew = [1_000, 10_000, 100_000, 1_000_000]
    bench_elementwise("Tensor + Tensor", py_add, "add", sizes_ew)
    bench_elementwise("Tensor - Tensor", py_sub, "sub", sizes_ew)
    bench_elementwise("Tensor * Tensor", py_mul, "mul", sizes_ew)

    sizes_mm = [(32, 64), (64, 256), (128, 512), (256, 1024), (1024, 1024)]
    bench_matmul(sizes_mm)

    print("\n" + "=" * 92)
    print(" 관전 포인트:")
    print("   - element-wise 는 대체로 50~200x. N 이 너무 작으면 호출 오버헤드,")
    print("     너무 크면 memory-bound 로 가속 줄어듦.")
    print("   - matmul 은 inner loop 가 작을 때 (cache 안 들어가는 크기) 더 큰")
    print("     가속. 큰 행렬은 memory-bound 로 수렴.")
    print("=" * 92)


if __name__ == "__main__":
    main()
