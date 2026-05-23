"""정적 ``range`` 로 반복하는 for 루프를 컴파일 타임에 펼친다.

Nadya 의 ``template<>`` 메타프로그래밍으로 SIMD 슬롯폭을 정해 펼치는 것을
Pydya 의 부분평가로 재현한다. ``for i in range(K)`` 의 ``K`` 가 fold 이후
정적 상수로 결정되면, 본문을 ``i = 0 .. K-1`` 로 펼쳐 그 자리에 풀어 둔다.

예::

    W = CompileVar('W')

    def dot_product(a, b):
        result = 0
        for i in range(W):
            result += a[i] * b[i]
        return result

``compile_source(src, env={'W': 4})`` 결과::

    def dot_product(a, b):
        result = 0
        result += a[0] * b[0]
        result += a[1] * b[1]
        result += a[2] * b[2]
        result += a[3] * b[3]
        return result
"""

from __future__ import annotations

import ast
import copy
from typing import List, Optional, Tuple

from pydya.passes.fold import fold

# 펼치기로 결정하는 최대 반복 횟수. 너무 큰 범위까지 펼치면 코드가 폭발하므로
# 보수적 기본값을 둔다(추후 attr opt-in 으로 강제 펼치기 가능).
DEFAULT_THRESHOLD = 64


def _range_args(call: ast.Call) -> Optional[Tuple[int, int, int]]:
    """``range(...)`` 호출에서 ``(start, stop, step)`` 을 추출. 정적 상수 정수가
    아니거나 step 이 0 이면 ``None``.
    """
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and not call.keywords
    ):
        return None
    args = call.args
    if not (1 <= len(args) <= 3):
        return None
    values: List[int] = []
    for arg in args:
        if not (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
        ):
            return None
        values.append(arg.value)
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    else:
        start, stop, step = values
    if step == 0:
        return None
    return start, stop, step


class _LocalLoopWalker(ast.NodeVisitor):
    """현재 for 루프 본문에 한정해서 ``break``/``continue``/loop_var Store 를
    탐지한다. 안쪽의 for/while/함수 정의로는 내려가지 않는다.
    """

    def __init__(self, loop_var: str):
        self.loop_var = loop_var
        self.has_break_or_continue = False
        self.stores_loop_var = False

    # 안쪽 루프/함수 경계 안의 break/continue 는 우리 for 와 무관하므로 무시.
    def visit_For(self, node):  # noqa: N802
        pass

    def visit_AsyncFor(self, node):  # noqa: N802
        pass

    def visit_While(self, node):  # noqa: N802
        pass

    def visit_FunctionDef(self, node):  # noqa: N802
        pass

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        pass

    def visit_Lambda(self, node):  # noqa: N802
        pass

    def visit_Break(self, node):  # noqa: N802
        self.has_break_or_continue = True

    def visit_Continue(self, node):  # noqa: N802
        self.has_break_or_continue = True

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Store) and node.id == self.loop_var:
            self.stores_loop_var = True


def _can_unroll(node: ast.For, count: int, threshold: int) -> bool:
    if count < 0 or count > threshold:
        return False
    if node.orelse:  # for-else 는 보수적으로 보류
        return False
    if not isinstance(node.target, ast.Name):
        return False
    walker = _LocalLoopWalker(node.target.id)
    for stmt in node.body:
        walker.visit(stmt)
    if walker.has_break_or_continue or walker.stores_loop_var:
        return False
    return True


def _unroll_once(body: List[ast.stmt], loop_var: str, value: int) -> List[ast.stmt]:
    """본문을 한 번 복제해 ``loop_var`` 를 ``value`` 로 치환·폴딩한 결과."""
    block = ast.Module(body=copy.deepcopy(body), type_ignores=[])
    fold(block, {loop_var: value})
    return block.body


class _Unroller(ast.NodeTransformer):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def visit_For(self, node: ast.For):  # noqa: N802
        # 먼저 안쪽 본문/iter 를 처리해서 중첩된 for 가 이미 펼쳐진 상태로 본다.
        self.generic_visit(node)

        triple = _range_args(node.iter)
        if triple is None:
            return node
        start, stop, step = triple
        values = list(range(start, stop, step))
        if not _can_unroll(node, len(values), self.threshold):
            return node

        if not values:
            # range(0) 등 빈 반복: 부모 본문이 비지 않도록 Pass 로 대체.
            return ast.copy_location(ast.Pass(), node)
        loop_var = node.target.id
        unrolled: List[ast.stmt] = []
        for value in values:
            unrolled.extend(_unroll_once(node.body, loop_var, value))
        return unrolled  # 부모 문장 리스트에 펼쳐 넣어진다


def unroll(tree: ast.AST, threshold: int = DEFAULT_THRESHOLD) -> ast.AST:
    """정적 ``range`` 로 반복하는 for 루프를 펼친다.

    ``threshold`` 보다 큰 반복 횟수는 코드 폭발을 막기 위해 펼치지 않는다.
    """
    new_tree = _Unroller(threshold).visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree
