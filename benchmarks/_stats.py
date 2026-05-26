"""벤치마크용 통계 측정 유틸리티.

각 벤치마크 스크립트가 공통으로 쓰는 측정 도구:

* :func:`time_samples` — 워밍업 후 반복 측정해 per-iter 시간 샘플 리스트 반환
* :func:`Stats` — 샘플에서 min/max/p50/p95/p99/mean/std/CoV 계산
* :func:`print_row` — 결과 한 줄 정리해 출력
* :func:`print_header` — 컬럼 헤더 출력

단순 평균 한 줄이 아니라 *분포* 를 본다 — GC/스케줄러/캐시 변동 시 p99 tail
이 흔들리는지, std 가 어느 정도인지 까지 한눈에 비교 가능.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List


def time_samples(
    fn: Callable[[], object],
    iters: int,
    warmup: int = 3,
    inner: int = 1,
) -> List[float]:
    """``fn`` 을 ``iters`` 회 측정. 각 측정은 ``inner`` 회 호출의 평균.

    워밍업 ``warmup`` 회 후 본 측정. 너무 빠른 fn (us 미만) 의 경우 ``inner`` 를
    높여 perf_counter 해상도 한계를 넘김 (예: inner=1000 이면 한 측정 = 1000회).
    반환 단위는 *초/호출*.
    """
    for _ in range(warmup):
        fn()
    out: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        elapsed = (time.perf_counter() - start) / inner
        out.append(elapsed)
    return out


class Stats:
    """샘플 리스트로부터 분포 통계 계산."""

    def __init__(self, samples: List[float]):
        if not samples:
            raise ValueError("samples is empty")
        s = sorted(samples)
        n = len(s)
        self.n = n
        self.min = s[0]
        self.max = s[-1]
        self.p50 = s[n // 2]
        self.p95 = s[min(n - 1, int(n * 0.95))]
        self.p99 = s[min(n - 1, int(n * 0.99))]
        self.mean = sum(s) / n
        self.std = math.sqrt(sum((x - self.mean) ** 2 for x in s) / n)
        # 변동계수(coefficient of variation) — std / mean. 측정 안정성 지표.
        self.cov = self.std / self.mean if self.mean > 0 else 0.0


def fmt_time(seconds: float) -> str:
    """초 단위 시간을 자동 단위(s/ms/us/ns) 로 5자리 폭에 맞춰 포맷."""
    if seconds >= 1.0:
        return f"{seconds:7.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:7.3f}ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:7.2f}us"
    return f"{seconds * 1e9:7.1f}ns"


def print_header(extra: str = "") -> None:
    cols = ["label", "min", "p50", "mean", "p95", "p99", "max", "std", "CoV"]
    widths = [28, 9, 9, 9, 9, 9, 9, 9, 7]
    line = "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    print(line + ("  " + extra if extra else ""))
    print("─" * len(line) + ("──" + "─" * len(extra) if extra else ""))


def print_row(label: str, stats: Stats, extra: str = "") -> None:
    print(
        f"{label:<28}  "
        f"{fmt_time(stats.min):>9}  "
        f"{fmt_time(stats.p50):>9}  "
        f"{fmt_time(stats.mean):>9}  "
        f"{fmt_time(stats.p95):>9}  "
        f"{fmt_time(stats.p99):>9}  "
        f"{fmt_time(stats.max):>9}  "
        f"{fmt_time(stats.std):>9}  "
        f"{stats.cov * 100:>6.2f}%"
        + ("  " + extra if extra else "")
    )


def print_speedup(baseline: Stats, current: Stats, label: str = "speedup") -> None:
    """두 stats 의 median(p50) 기준 가속비 + CI 근사 출력."""
    ratio_med = baseline.p50 / current.p50 if current.p50 > 0 else float("inf")
    ratio_mean = baseline.mean / current.mean if current.mean > 0 else float("inf")
    print(
        f"  → {label}: median {ratio_med:5.2f}x   mean {ratio_mean:5.2f}x"
    )
