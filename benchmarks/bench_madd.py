"""표현식 융합 ``a*b+c`` (madd) — 미융합 vs 융합.

목적: alloc 1회 + 메모리 1패스로 줄이는 효과가 *어디서 가장 잘 나타나는지*
정량. 메모리-바운드 element-wise op 의 체인에서 fusion 이 이론적으로
~33% 메모리 traffic 감소를 줄 수 있는데, 실측이 그에 얼마나 근접하는지.

추가로 분포(p99/std) 까지 봐서 *융합본이 jitter 가 작은지* (alloc 변동성
제거) 확인.

실행:  PYTHONPATH=. python benchmarks/bench_madd.py
"""

from __future__ import annotations

import os
import random
import sys

from pydya import Tensor
from pydya._tensor import madd

from _stats import Stats, fmt_time, time_samples

random.seed(0)


def main():
    print("=" * 96)
    print(" 표현식 융합 a*b+c — UNFUSED (Tensor 연산자 체인) vs FUSED (madd)")
    print("=" * 96)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print()
    print(f"{'N':>10}    {'UNFUSED p50':>14}    {'FUSED p50':>14}    {'speedup':>9}    "
          f"{'unfused std/p99':>22}    {'fused std/p99':>22}")
    print("─" * 96)

    sizes = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
    for n in sizes:
        a = Tensor([random.random() for _ in range(n)])
        b = Tensor([random.random() for _ in range(n)])
        c = Tensor([random.random() for _ in range(n)])

        inner = max(1, 200_000 // n)
        iters = 60

        samples_unfused = time_samples(lambda: a * b + c, iters=iters, inner=inner)
        samples_fused = time_samples(lambda: madd(a, b, c), iters=iters, inner=inner)
        su = Stats(samples_unfused)
        sf = Stats(samples_fused)
        speedup = su.p50 / sf.p50

        print(
            f"{n:>10}    "
            f"{fmt_time(su.p50):>14}    "
            f"{fmt_time(sf.p50):>14}    "
            f"{speedup:>7.2f}x    "
            f"{fmt_time(su.std)} / {fmt_time(su.p99)}    "
            f"{fmt_time(sf.std)} / {fmt_time(sf.p99)}"
        )

    print()
    print("=" * 96)
    print(" 관전 포인트:")
    print("   - 작은 N: 호출 오버헤드 비중이 커서 fusion 효과 작음")
    print("   - 중간 N (cache 들어가는 크기): 임시 alloc + 추가 메모리 패스 제거로")
    print("     이론적 ~1.5x 근접")
    print("   - 큰 N (cache 초과): memory bandwidth 가 천장이라 fusion 효과 유지")
    print("   - jitter: fused 의 p99/std 가 unfused 보다 작아야 — 임시 alloc 의")
    print("     변동성이 사라지기 때문")
    print("=" * 96)


if __name__ == "__main__":
    main()
