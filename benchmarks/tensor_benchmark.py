"""순수 파이썬 list 와 C 레벨 Tensor 의 동일 연산 성능 비교.

같은 element-wise 산술을 두 가지로 수행한다:
  - LIST  : list comprehension (파이썬 인터프리터 루프)
  - TENSOR: pydya.Tensor (C 핫루프 + GIL 해제 + 자동 벡터화)

목적은 Phase 2 의 가치를 정량화하는 것 — Phase 1 unroll 이 못 주는
런타임 가속이 *C 레벨 본문* 에서 어디서 나오는지 보인다.

실행:
    PYTHONPATH=. python benchmarks/tensor_benchmark.py
"""

import os
import sys
import time

from pydya import Tensor

assert Tensor is not None, "C 확장이 빌드되지 않았습니다 (python setup.py build_ext --inplace)."

N = 1_000_000

# 동일 데이터를 list 와 Tensor 양쪽에 준비
src_a = [float(i) for i in range(N)]
src_b = [float(i) * 0.5 for i in range(N)]

list_a = src_a
list_b = src_b
tensor_a = Tensor(src_a)
tensor_b = Tensor(src_b)


def bench(label, fn):
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    return label, elapsed, out


def list_mul():
    return [a * b for a, b in zip(list_a, list_b)]


def list_mul_add():
    return [a * b + a for a, b in zip(list_a, list_b)]


def tensor_mul():
    return tensor_a * tensor_b


def tensor_mul_add():
    return tensor_a * tensor_b + tensor_a


def main():
    print("=" * 64)
    print(" C 레벨 Tensor vs 파이썬 list — element-wise 산술")
    print("=" * 64)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print(f"  N      : {N:,}")
    print("-" * 64)

    cases = [
        ("list[float]    : a * b           ", list_mul),
        ("Tensor (C)     : a * b           ", tensor_mul),
        ("list[float]    : a * b + a       ", list_mul_add),
        ("Tensor (C)     : a * b + a       ", tensor_mul_add),
    ]

    times = {}
    for label, fn in cases:
        # warm-up 1회
        fn()
        _, t, _ = bench(label, fn)
        times[label] = t
        print(f"  {label}: {t:.3f}s")

    print("-" * 64)
    s1 = times["list[float]    : a * b           "]
    t1 = times["Tensor (C)     : a * b           "]
    s2 = times["list[float]    : a * b + a       "]
    t2 = times["Tensor (C)     : a * b + a       "]
    print(f"  speedup (a*b)        : {s1 / t1:.2f}x")
    print(f"  speedup (a*b + a)    : {s2 / t2:.2f}x")
    print("=" * 64)


if __name__ == "__main__":
    main()
