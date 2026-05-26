"""attr[parallel] — 워크로드 크기 / 워커 수 별 가속 분석.

목적: 우리 병렬 백엔드 (3.14 서브인터프리터 / 3.11 스레드풀) 의 손익분기점
가시화. 너무 가벼운 워크로드는 dispatch 비용에 묻히고, 충분히 무거운 본문
+ 적절한 워커수에서 코어 수에 가까운 가속.

이 벤치는 *직렬 baseline 대비 N-워커 가속비* 를 per-item work × N-items
조합 별로 측정해, *"언제 attr[parallel] 을 붙일 가치가 있는가"* 의
경험적 가이드를 만든다.

실행:  PYTHONPATH=. python benchmarks/bench_parallel.py
"""

from __future__ import annotations

import os
import sys

import pydya.runtime as rt

from _stats import Stats, fmt_time, time_samples


def run_config(expr, n_items, workers_list, inner, iters=15):
    """동일 expr·N_items 로 워커수만 바꿔가며 시간 측정."""
    print(f"\n  expr            : {expr}")
    print(f"  N items         : {n_items},   inner repeats: {inner},  iters: {iters}")
    backend = "subinterpreter" if rt._subinterpreter_runner() is not None else "threadpool"
    print(f"  backend         : {backend}")
    print(f"  {'workers':>8}    {'p50':>10}    {'mean ±std':>22}    {'p99':>10}    {'speedup vs w=1':>16}")
    print("  " + "─" * 80)

    base_p50 = None
    for w in workers_list:
        target = [0] * n_items
        samples = time_samples(
            lambda: rt.parallel_map_into(target, range(n_items), expr, "i", {}, workers=w),
            iters=iters, inner=inner,
        )
        s = Stats(samples)
        if base_p50 is None:
            base_p50 = s.p50
        speedup = base_p50 / s.p50 if s.p50 > 0 else float("inf")
        print(
            f"  {w:>8}    "
            f"{fmt_time(s.p50):>10}    "
            f"{fmt_time(s.mean)} ±{fmt_time(s.std):<8}    "
            f"{fmt_time(s.p99):>10}    "
            f"{speedup:>14.2f}x"
        )


def main():
    cpus = os.cpu_count() or 1
    backend = "subinterpreter (3.14+)" if rt._subinterpreter_runner() is not None \
              else "threadpool (3.11/3.12/3.13)"

    print("=" * 90)
    print(" attr[parallel] — 워크로드 / 워커수 별 가속 분석")
    print("=" * 90)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {cpus}")
    print(f"  backend: {backend}")

    workers_list = [1, 2, 4, cpus] if cpus > 4 else [1, 2, 4]
    workers_list = sorted(set(workers_list))

    # 워크로드 1: 가벼운 본문 (호출 오버헤드가 큰 영역) — 병렬화 무의미해야 함
    print("\n## 워크로드 A: 가벼운 본문 (스칼라 한 줄)")
    run_config("i * i", n_items=64, workers_list=workers_list, inner=200)

    # 워크로드 2: 중간 본문
    print("\n## 워크로드 B: 중간 본문 (간단한 누적)")
    run_config("sum((i * j) % 7 for j in range(10000))",
               n_items=32, workers_list=workers_list, inner=5)

    # 워크로드 3: 무거운 본문 (병렬화 효과 확실)
    print("\n## 워크로드 C: 무거운 본문 (긴 inner loop)")
    run_config("sum((i * j) % 7 for j in range(2000000))",
               n_items=8, workers_list=workers_list, inner=1, iters=5)

    print()
    print("=" * 90)
    print(" 관전 포인트:")
    print("   - 워크로드 A (가벼움): 워커 증가가 거의 의미 없거나 오히려 손해.")
    print("     dispatch/marshaling 비용 > 본문 비용.")
    print("   - 워크로드 B (중간): 워커 2~코어수 에서 의미 있는 가속, 백엔드 의존적.")
    print("   - 워크로드 C (무거움): 가장 깨끗한 코어수-비례 가속.")
    print("     3.14 서브인터프리터에서 거의 선형, 3.11 스레드풀은 GIL 로 ~1x.")
    print("=" * 90)


if __name__ == "__main__":
    main()
