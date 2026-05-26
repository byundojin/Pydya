"""attr[unroll] — Python 레벨에서 *실제로* 효과가 나는 워크로드 측정.

unroll 의 Python-level 효과 출처는 한정적:
  (1) FOR_ITER / STORE_FAST 옵코드 K 회 제거
  (2) 본문에 ``i`` 가 들어가는 *분기/비교* 가 컴파일 타임에 상수로 접혀 사라짐
  (3) ``i`` 기반 표현식이 literal 로 fold 되어 추가 dce 가능

(1) 단독으론 효과 거의 0 (CPython 의 FOR_ITER 가 워낙 빠름). (2)(3) 가
크게 작용하는 워크로드 — 즉 *분기 많은 작은 루프* — 에서만 의미가 나타남.

이 벤치마크는 그런 워크로드 (i 의존 분기 + 매개변수 fold) 에서 unroll 이
얼마나 가속을 주는지 정량화한다. *일반적인 Python 루프에 unroll 을 붙인다고
빨라지는 게 아니라*, 부분평가가 본문을 더 단순화할 수 있는 경우에 한해
의미가 있다는 사실의 정량적 증거.

실행:  PYTHONPATH=. python benchmarks/bench_unroll.py
"""

from __future__ import annotations

import os
import random
import sys

from pydya import compile_source

from _stats import Stats, fmt_time, time_samples

random.seed(0)


# ─── 워크로드 1: i 의존 분기 (piecewise activation) ──────────────────────
#
# 본문에 i 와 THRESH 의 비교가 있어 unroll 시 분기 전체가 사라진다.

PIECEWISE_SRC_BASE = """
from pydya import attr
K = CompileVar('K')
THRESH = CompileVar('THRESH')

def piecewise(x):
    out = [0.0] * K
    {marker}
    for i in range(K):
        v = x[i]
        if i < THRESH:
            out[i] = v if v > 0.0 else 0.0
        else:
            out[i] = v * 0.5 if v > 0.0 else v * 0.2
    return out
"""

PIECEWISE_NO_UNROLL = PIECEWISE_SRC_BASE.format(marker="")
PIECEWISE_UNROLL = PIECEWISE_SRC_BASE.format(marker="attr[{'unroll': True}]")


# ─── 워크로드 2: i 의존 계수 (가중합) ───────────────────────────────────
#
# 본문이 i 자체를 산술에 쓰며 계수가 i 에 따라 결정. unroll 시 i 가 literal
# 이라 산술이 일부 fold 됨.

WSUM_SRC_BASE = """
from pydya import attr
K = CompileVar('K')

def wsum(x):
    s = 0.0
    {marker}
    for i in range(K):
        if i % 3 == 0:
            s += x[i] * 1.5
        elif i % 3 == 1:
            s += x[i] * 0.7
        else:
            s -= x[i] * 0.3
    return s
"""

WSUM_NO_UNROLL = WSUM_SRC_BASE.format(marker="")
WSUM_UNROLL = WSUM_SRC_BASE.format(marker="attr[{'unroll': True}]")


def compile_and_get(src, env, name):
    compiled = compile_source(src, env=env)
    ns = {}
    exec(compiled, ns)
    return ns[name], compiled


def run_one(name, src_off, src_on, env, fn_name, make_input, label_off, label_on):
    fn_off, code_off = compile_and_get(src_off, env, fn_name)
    fn_on, code_on = compile_and_get(src_on, env, fn_name)
    x = make_input()
    iters = 200
    inner = 200
    s_off = Stats(time_samples(lambda: fn_off(x), iters=iters, inner=inner))
    s_on = Stats(time_samples(lambda: fn_on(x), iters=iters, inner=inner))
    speedup = s_off.p50 / s_on.p50 if s_on.p50 > 0 else float("inf")
    print(f"\n## {name}")
    print(f"  env: {env}")
    print(f"  [{label_off}] (no unroll) compiled:")
    for line in code_off.strip().splitlines()[:8]:
        print(f"      {line}")
    print(f"  [{label_on}] (with attr[unroll]) compiled:")
    for line in code_on.strip().splitlines()[:8]:
        print(f"      {line}")
    if len(code_on.strip().splitlines()) > 8:
        print(f"      ... (+{len(code_on.strip().splitlines()) - 8} more lines)")
    print()
    print(f"  {label_off:<28}  p50 {fmt_time(s_off.p50)}  mean {fmt_time(s_off.mean)} ±{fmt_time(s_off.std)}  p99 {fmt_time(s_off.p99)}")
    print(f"  {label_on:<28}  p50 {fmt_time(s_on.p50)}  mean {fmt_time(s_on.mean)} ±{fmt_time(s_on.std)}  p99 {fmt_time(s_on.p99)}")
    print(f"  → median speedup: {speedup:.2f}x")


def main():
    print("=" * 100)
    print(" attr[unroll] — Python 레벨 가속이 실제로 나타나는 워크로드")
    print("=" * 100)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")

    run_one(
        "워크로드 1: piecewise activation (i 의존 분기)",
        PIECEWISE_NO_UNROLL,
        PIECEWISE_UNROLL,
        env={"K": 32, "THRESH": 16},
        fn_name="piecewise",
        make_input=lambda: [random.uniform(-1, 1) for _ in range(32)],
        label_off="serial loop + per-iter branch",
        label_on="unrolled, branches resolved",
    )
    run_one(
        "워크로드 2: i-modular 가중합",
        WSUM_NO_UNROLL,
        WSUM_UNROLL,
        env={"K": 30},
        fn_name="wsum",
        make_input=lambda: [random.uniform(-1, 1) for _ in range(30)],
        label_off="serial loop + i%3 분기",
        label_on="unrolled, i%3 도 상수 fold",
    )

    print("\n" + "=" * 100)
    print(" 관전 포인트:")
    print("   - 'i 의존 분기' 가 있는 작은 루프에서만 unroll 이 의미 있는 가속을 줌")
    print("     — 분기 비교/JUMP 옵코드가 사라지고 본문이 직선 코드가 되기 때문.")
    print("   - C 안에 핫코드가 있는 워크로드(텐서 추론 등)엔 attr[unroll] 의")
    print("     runtime 기여 0. unroll 은 *부분평가 substrate* 로 부분적 의미.")
    print("   - 즉 unroll 은 일반 가속기가 아니라 *부분평가가 의미 있는 케이스에서만")
    print("     도움이 되는* 한정 도구다.")
    print("=" * 100)


if __name__ == "__main__":
    main()
