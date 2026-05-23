"""``attr[{'unroll': True}]`` 마커가 붙은 for 루프를 컴파일 타임에 펼친다.

소스에서 for 루프 바로 앞에 ``attr[{'unroll': True}]`` 를 두면, Pydya 가 그
다음 ``for i in range(K)`` 의 본문을 ``i = 0..K-1`` 로 펼친다. ``K`` 가
컴파일 타임에 상수로 결정되어야 하며(예: ``range(W)`` 에서 ``W`` 가
:class:`CompileVar`), 그렇지 않으면 펼치지 않는다.

마커 인식·소비는 :mod:`pydya.passes.parallelize` 가 일괄 담당하고, 이 패스는
거기서 :data:`_UNROLL_FLAG` 가 표시된 for 노드만 처리한다.

한계 (정직한 경고)
------------------
이 unroll 은 *부분평가 컴파일러의 substrate* 다. 펼친 결과는 인간이 읽을 수
있는 직선 Python 이고, 다음과 같은 가치를 가진다:

* 본문에 다른 ``CompileVar`` 가 있으면 그 자리에 상수가 박혀 추가 fold 가능
* (예정) Phase 2 C Tensor / Phase 3 표현식 융합 패스가 직선 트리에서 패턴
  매칭하기 쉬워짐

그러나 **그 자체로는 런타임 성능 이득이 없다.** Optimium 이 unroll 로 노리는
이득(메모리 접근 감소, SIMD 벡터화, ILP, register reuse)은 모두 *네이티브
바이너리 컴파일* 을 전제로 한다. CPython 바이트코드 VM 위에서는 펼친 코드가
오히려 코드 크기 증가로 손해를 볼 수 있다. 진짜 가속은 Phase 2/3 에서 본문을
네이티브 C 호출로 lowering 한 뒤에야 따라온다.

예::

    W = CompileVar('W')

    def dot_product(a, b):
        result = 0
        attr[{'unroll': True}]
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
# 보수적 기본값을 둔다.
DEFAULT_THRESHOLD = 64

# parallelize 패스가 attr[{'unroll': True}] 마커를 읽고 다음 for 노드에 다는
# 속성 이름. unroll 패스는 이 플래그가 붙은 노드만 처리한다.
_UNROLL_FLAG = "_pydya_unroll"


class UnrollError(Exception):
    """unroll opt-in 이 잘못 사용된 경우."""


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


def _safe_to_unroll(node: ast.For) -> bool:
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
        # 먼저 안쪽 본문을 처리해서 중첩된 marked for 가 이미 펼쳐진 상태로 본다.
        self.generic_visit(node)

        # attr opt-in 이 없는 for 는 손대지 않는다.
        if not getattr(node, _UNROLL_FLAG, False):
            return node

        triple = _range_args(node.iter)
        if triple is None:
            raise UnrollError(
                "attr[{'unroll': True}] 는 range(...) 인자가 모두 컴파일 타임 "
                "상수일 때만 펼칠 수 있습니다(CompileVar 로 바인딩하거나 리터럴)."
            )
        if not _safe_to_unroll(node):
            raise UnrollError(
                "attr[{'unroll': True}] 가 붙은 for 의 본문이 안전하지 않습니다 "
                "(break/continue/for-else/루프변수 Store 중 하나)."
            )

        start, stop, step = triple
        values = list(range(start, stop, step))
        if len(values) > self.threshold:
            raise UnrollError(
                f"펼침 횟수 {len(values)} 가 임계값 {self.threshold} 를 초과합니다."
            )

        if not values:
            # range(0) 등 빈 반복: 부모 본문이 비지 않도록 Pass 로 대체.
            return ast.copy_location(ast.Pass(), node)
        loop_var = node.target.id
        unrolled: List[ast.stmt] = []
        for value in values:
            unrolled.extend(_unroll_once(node.body, loop_var, value))
        return unrolled  # 부모 문장 리스트에 펼쳐 넣어진다


def unroll(tree: ast.AST, threshold: int = DEFAULT_THRESHOLD) -> ast.AST:
    """``attr[{'unroll': True}]`` 로 표시된 for 루프를 펼친다.

    표시는 :mod:`pydya.passes.parallelize` 가 attr 마커를 처리하면서 단다.
    """
    new_tree = _Unroller(threshold).visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree
