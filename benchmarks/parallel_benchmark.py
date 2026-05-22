"""attr 병렬 map 의 직렬 vs 병렬 성능을 측정한다.

실행:
    PYTHONPATH=. python benchmarks/parallel_benchmark.py          # 현재 파이썬
    PYTHONPATH=. <python3.14> benchmarks/parallel_benchmark.py    # 서브인터프리터

순수 파이썬 본문(캡처 없음)을 일부러 무겁게 만들어 백엔드별 멀티코어
효과를 본다. 3.14+ 에서는 서브인터프리터(인터프리터별 GIL)로 진짜 병렬,
그 이하에서는 스레드풀이라 순수 파이썬 본문은 가속되지 않는다(정확성만 보장).
"""

import os
import sys
import time

import pydya.runtime as rt

# 캡처가 없어야 서브인터프리터가 __main__ 을 재실행하지 않는다.
EXPR = "sum((i * j) % 7 for j in range(3000000))"
LOOP_VAR = "i"
N = 8


def _bench(workers):
    target = [0] * N
    start = time.perf_counter()
    rt.parallel_map_into(target, range(N), EXPR, LOOP_VAR, {}, workers=workers)
    return time.perf_counter() - start, target


def main():
    backend = (
        "subinterpreter"
        if rt._subinterpreter_runner() is not None
        else "threadpool"
    )
    cpus = os.cpu_count() or 1
    print(f"python   : {sys.version.split()[0]}")
    print(f"cpus     : {cpus}")
    print(f"backend  : {backend}")
    print(f"items(N) : {N}")

    serial_t, serial_r = _bench(workers=1)
    par_t, par_r = _bench(workers=cpus)
    assert serial_r == par_r, "병렬 결과가 직렬과 다릅니다!"

    print(f"\nserial   (workers=1)    : {serial_t:.3f}s")
    print(f"parallel (workers={cpus})    : {par_t:.3f}s")
    print(f"speedup                 : {serial_t / par_t:.2f}x")
    if backend == "threadpool":
        print(
            "\n[참고] 스레드풀 백엔드라 순수 파이썬 본문은 GIL 로 직렬화됩니다. "
            "진짜 멀티코어는 Python 3.14+ 서브인터프리터에서 나옵니다."
        )


if __name__ == "__main__":
    main()
