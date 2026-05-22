"""`attr[{'parallel': True}]` 를 붙였을 때와 안 붙였을 때를 비교하는 벤치마크.

똑같은 for 루프를 두 가지로 **컴파일**해서 실행 시간을 잰다.

  - PLAIN : attr 없음            -> 직렬 잔여 코드
  - ATTR  : attr[{'parallel'..}] -> 병렬 실행 코드 (parallel_map_into)

즉 "어노테이션 한 줄을 붙이는 것만으로 얼마나 빨라지는가" 를 측정한다.

실행:
    PYTHONPATH=. python benchmarks/parallel_benchmark.py          # 현재 파이썬
    PYTHONPATH=. <python3.14> benchmarks/parallel_benchmark.py    # 서브인터프리터
"""

import os
import sys
import time

import pydya.runtime as rt
from pydya import compile_source

N = 8
WORKERS = os.cpu_count() or 1

# 무거운 순수 파이썬 본문 (캡처가 없어야 서브인터프리터가 __main__ 을 재실행하지 않음)
BODY = "out[i] = sum((i * j) % 7 for j in range(3000000))"

PLAIN_SRC = f"""
out = [0] * {N}
for i in range({N}):
    {BODY}
"""

ATTR_SRC = f"""
from pydya import attr
out = [0] * {N}
attr[{{'parallel': True, 'workers': {WORKERS}}}]
for i in range({N}):
    {BODY}
"""


def _measure(src):
    """``src`` 를 컴파일·실행하고 (실행시간, 결과, 컴파일된 소스) 를 반환."""
    compiled = compile_source(src)
    namespace = {}
    start = time.perf_counter()
    exec(compiled, namespace)
    return time.perf_counter() - start, namespace["out"], compiled


def _show(title, compiled, elapsed):
    print(f"  [{title}]")
    for line in compiled.strip().splitlines():
        print(f"      {line}")
    print(f"    -> {elapsed:.3f}s\n")


def main():
    backend = (
        "subinterpreter (3.14+, 진짜 멀티코어)"
        if rt._subinterpreter_runner() is not None
        else "threadpool (GIL 로 순수 파이썬 본문은 직렬화됨)"
    )

    print("=" * 64)
    print(" attr 병렬 어노테이션 효과 벤치마크")
    print("=" * 64)
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  cpus    : {os.cpu_count()}")
    print(f"  backend : {backend}")
    print(f"  본문    : {BODY}")
    print(f"  N       : {N}, workers : {WORKERS}")
    print("-" * 64)

    plain_t, plain_r, plain_c = _measure(PLAIN_SRC)
    attr_t, attr_r, attr_c = _measure(ATTR_SRC)

    _show("PLAIN  (attr 없음, 직렬)", plain_c, plain_t)
    _show(f"ATTR   (attr[parallel], 병렬)", attr_c, attr_t)

    print("-" * 64)
    print(f"  결과 동일 : {plain_r == attr_r}")
    print(f"  speedup   : {plain_t / attr_t:.2f}x  (PLAIN {plain_t:.3f}s -> ATTR {attr_t:.3f}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
