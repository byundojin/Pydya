"""병렬화된 코드가 호출하는 런타임 헬퍼.

``attr[{'parallel': True}]`` 로 표시된 독립 반복 map(``out[i] = expr``)을
컴파일러가 :func:`parallel_map_into` 호출로 바꾼다. 컴파일러는 클로저 대신
**expr 의 소스 문자열 + 루프 변수명 + 캡처 값(dict)** 을 넘긴다. 그래야 각
백엔드가 커널을 동일하게 재구성할 수 있고, 특히 서브인터프리터로 실어보낼
수 있다.

실행 백엔드 (자동 선택)
-----------------------
1. **서브인터프리터** (``concurrent.interpreters``, Python 3.14+): 인터프리터별
   GIL(PEP 684) 덕에 순수 파이썬 본문도 진짜 멀티코어로 돈다. 캡처/결과가
   공유 불가(pickle 불가)하면 자동으로 스레드풀로 폴백한다.
2. **스레드풀**: 모든 버전에서 동작. GIL 을 푸는 작업(numpy 등)만 실제
   가속되고 순수 파이썬 본문은 직렬화되지만 결과는 항상 정확하다.
3. **직렬**: 워커 1개이거나 항목이 하나일 때.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Tuple


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


def _make_kernel(expr_src: str, loop_var: str, captures: Dict[str, Any]):
    """``expr`` 을 계산하는 ``_k(loop_var)`` 함수를 만든다. 캡처는 전역으로 둔다."""
    namespace = dict(captures)
    exec(f"def _k({loop_var}):\n    return {expr_src}", namespace)
    return namespace["_k"]


def parallel_map_into(
    target: Any,
    iterable: Iterable[Any],
    expr_src: str,
    loop_var: str,
    captures: Dict[str, Any],
    workers: int | None = None,
) -> Any:
    """``for i in iterable: target[i] = <expr>`` 를 병렬로 수행하고 target 반환.

    각 반복은 서로 다른 슬롯(``target[i]``)에만 쓰고 본문이 ``target`` 을 읽지
    않음이 컴파일 타임에 보장되므로 반복 순서와 무관하게 안전하다.
    """
    indices = list(iterable)
    if not indices:
        return target
    n = workers if workers else _default_workers()
    if n <= 1 or len(indices) == 1:
        kernel = _make_kernel(expr_src, loop_var, captures)
        for i in indices:
            target[i] = kernel(i)
        return target
    for i, value in _parallel(expr_src, loop_var, captures, indices, n):
        target[i] = value
    return target


def _parallel(
    expr_src: str,
    loop_var: str,
    captures: Dict[str, Any],
    indices: List[Any],
    workers: int,
) -> List[Tuple[Any, Any]]:
    chunks = _chunk(indices, workers)
    runner = _subinterpreter_runner()
    if runner is not None:
        try:
            return runner(expr_src, loop_var, captures, chunks)
        except Exception:
            # 공유 불가 캡처/결과, import 불가 등 → 스레드풀로 폴백.
            pass
    return _thread_runner(expr_src, loop_var, captures, chunks)


def _thread_runner(expr_src, loop_var, captures, chunks):
    kernel = _make_kernel(expr_src, loop_var, captures)

    def work(chunk):
        return [(i, kernel(i)) for i in chunk]

    out: List[Tuple[Any, Any]] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        for pairs in ex.map(work, chunks):
            out.extend(pairs)
    return out


def _subinterpreter_runner() -> Callable | None:
    try:
        from concurrent.futures import InterpreterPoolExecutor
    except ImportError:
        return None

    def run(expr_src, loop_var, captures, chunks):
        out: List[Tuple[Any, Any]] = []
        with InterpreterPoolExecutor(max_workers=len(chunks)) as ex:
            futures = [
                ex.submit(_subinterpreter_worker, expr_src, loop_var, captures, chunk)
                for chunk in chunks
            ]
            for future in futures:
                out.extend(future.result())
        return out

    return run


def _subinterpreter_worker(expr_src, loop_var, captures, chunk):
    """서브인터프리터에서 실행되는 워커. 모듈 최상위라 pickle 참조가 가능하다."""
    kernel = _make_kernel(expr_src, loop_var, captures)
    return [(i, kernel(i)) for i in chunk]
