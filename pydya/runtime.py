"""병렬화된 코드가 호출하는 런타임 헬퍼.

``attr[{'parallel': True}]`` 로 표시된 독립 반복 map(``out[i] = expr``)을
컴파일러가 :func:`parallel_map_into` 호출로 바꾼다.

실행 백엔드에 대하여
--------------------
진짜 멀티코어로 순수 파이썬 본문을 돌리려면 인터프리터별 GIL(PEP 684, 3.12+)
또는 프리스레드 빌드가 필요하다. 매끄러운 서브인터프리터 풀 API 는 3.14
(``concurrent.interpreters``)부터다. 그 전 버전에서는 스레드풀로 실행하며,
이 경우 GIL 을 푸는 작업(numpy 등)만 실제로 병렬 가속되고 순수 파이썬 본문은
직렬화된다. 결과 자체는 어느 백엔드에서나 직렬 실행과 동일하다.

여기서는 청크 분할 스레드풀(+직렬 폴백)을 제공한다. 3.14+ 서브인터프리터
백엔드는 :func:`_compute` 를 교체하는 자리로 남겨 둔다.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional


def _chunk(seq: List[Any], n: int) -> List[List[Any]]:
    """``seq`` 를 최대 ``n`` 개의 연속 청크로 균등 분할한다."""
    n = max(1, min(n, len(seq)))
    size, rem = divmod(len(seq), n)
    chunks = []
    start = 0
    for k in range(n):
        end = start + size + (1 if k < rem else 0)
        chunks.append(seq[start:end])
        start = end
    return [c for c in chunks if c]


def _default_workers() -> int:
    return os.cpu_count() or 1


def parallel_map_into(
    target: Any,
    iterable: Iterable[Any],
    fn: Callable[[Any], Any],
    workers: Optional[int] = None,
) -> Any:
    """``for i in iterable: target[i] = fn(i)`` 를 병렬로 수행하고 target 을 반환.

    각 반복은 서로 다른 슬롯(``target[i]``)에만 쓰며 본문이 ``target`` 을 읽지
    않음이 컴파일 타임에 보장되므로, 반복 순서와 무관하게 안전하다.
    """
    indices = list(iterable)
    if not indices:
        return target
    n = workers if workers else _default_workers()
    if n <= 1 or len(indices) == 1:
        for i in indices:
            target[i] = fn(i)
        return target
    for i, value in _compute(fn, indices, n):
        target[i] = value
    return target


def _compute(fn, indices, workers):
    """``[(i, fn(i)), ...]`` 를 청크 단위 스레드풀로 계산한다."""
    chunks = _chunk(indices, workers)

    def run(chunk):
        return [(i, fn(i)) for i in chunk]

    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        for pairs in ex.map(run, chunks):
            yield from pairs
