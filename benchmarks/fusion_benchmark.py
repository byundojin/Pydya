"""미융합 ``a*b+c`` 와 융합 ``madd(a,b,c)`` 의 성능 비교.

같은 결과를 두 가지로 계산:
  - UNFUSED : 개별 연산자 (Phase 2). 임시 텐서 2개 할당, 메모리 추가 순회.
  - FUSED   : pydya._tensor.madd (Phase 3). 단일 할당, 단일 순회.

목적은 Phase 3 융합의 *증분* 가치를 정량화하는 것 — Phase 2 의 47x 위에
체인 표현식에서 추가로 얼마나 더 빨라지는가.

실행:  PYTHONPATH=. python benchmarks/fusion_benchmark.py
"""

import os
import sys
import time

from pydya import Tensor
from pydya._tensor import madd

N = 1_000_000
ITERS = 50

# 값이 [-1, 1] 안에 머물도록 — float32 dynamic range 가 결과 비교를 흐리지 않게.
a = Tensor([((i % 97) / 97.0) for i in range(N)])
b = Tensor([((i % 89) / 89.0) for i in range(N)])
c = Tensor([((i % 79) / 79.0) for i in range(N)])


def bench(label, fn):
    # warm-up
    fn()
    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
    elapsed = time.perf_counter() - start
    return label, elapsed, out


def unfused():
    return a * b + c


def fused():
    return madd(a, b, c)


def main():
    print("=" * 64)
    print(" UNFUSED a*b+c vs FUSED madd(a, b, c)")
    print("=" * 64)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print(f"  N      : {N:,}")
    print(f"  iters  : {ITERS}")
    print("-" * 64)

    _, t_unfused, r_unfused = bench("UNFUSED", unfused)
    _, t_fused, r_fused = bench("FUSED  ", fused)

    # 결과 정합성
    diff = max(
        abs(x - y) for x, y in zip(r_unfused.to_list(), r_fused.to_list())
    )
    print(f"  UNFUSED (a*b + c)     : {t_unfused:.3f}s")
    print(f"  FUSED   (madd(a,b,c)) : {t_fused:.3f}s")
    print(f"  speedup               : {t_unfused / t_fused:.2f}x")
    print(f"  max abs diff          : {diff:g}")
    print("=" * 64)


if __name__ == "__main__":
    main()
