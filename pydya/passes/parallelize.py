"""``attr[{...}]`` 마커 처리 일괄 담당 패스.

for 루프 바로 앞의 ``attr[{...}]`` 마커를 인식해, 옵션 키에 따라 다음과 같이
처리한다:

* ``{'parallel': True}`` — Nadya 의 ``attr[Parallel : true] for(...)`` 대응.
  **반복 간 파괴적 갱신이 없는 독립 map** 인지 보수적으로 검사한 뒤
  :func:`pydya.runtime.parallel_map_into` 호출로 lowering.
* ``{'unroll': True}`` — for 노드에 ``_UNROLL_FLAG`` 를 달아 이후
  :mod:`pydya.passes.unroll` 이 펼치도록 한다.

두 키가 동시에 참이면 충돌로 거부한다(같은 루프를 병렬 호출로 바꾸면서 동시에
펼치는 의미가 모호). 마커 자체는 항상 제거된다.
"""

from __future__ import annotations

import ast
import builtins
from typing import List, Optional, Tuple

from pydya.passes.unroll import _UNROLL_FLAG, UnrollError

_RT_ALIAS = "__pydya_rt"
_BODY_FIELDS = ("body", "orelse", "finalbody")
_BUILTIN_NAMES = frozenset(dir(builtins))


class UnsafeParallelLoop(Exception):
    """``attr[{'parallel': True}]`` 를 안전한 독립 map 으로 증명할 수 없을 때."""


def _is_attr_marker(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "attr"
    )


def _marker_options(node: ast.Expr) -> dict:
    opts = {}
    slc = node.value.slice  # type: ignore[attr-defined]
    if isinstance(slc, ast.Dict):
        for key, value in zip(slc.keys, slc.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                opts[key.value] = value
    return opts


def _is_truthy_const(node: Optional[ast.expr]) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


def _independent_map(
    for_node: ast.For,
) -> Optional[Tuple[str, str, ast.expr]]:
    """독립 map 이면 ``(loop_var, target_name, expr)`` 를, 아니면 None.

    조건: 단일 ``target[i] = expr`` 문, 인덱스가 정확히 루프 변수 ``i``,
    그리고 ``expr`` 안에서 ``target`` 을 읽지 않음(반복 간 의존 금지).
    """
    if for_node.orelse or not isinstance(for_node.target, ast.Name):
        return None
    if len(for_node.body) != 1:
        return None
    stmt = for_node.body[0]
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return None
    tgt = stmt.targets[0]
    if not (
        isinstance(tgt, ast.Subscript)
        and isinstance(tgt.value, ast.Name)
        and isinstance(tgt.slice, ast.Name)
    ):
        return None
    loop_var = for_node.target.id
    if tgt.slice.id != loop_var:
        return None
    target_name = tgt.value.id
    expr = stmt.value
    for sub in ast.walk(expr):
        if (
            isinstance(sub, ast.Name)
            and sub.id == target_name
            and isinstance(sub.ctx, ast.Load)
        ):
            return None
    return loop_var, target_name, expr


def _free_names(expr: ast.expr, loop_var: str) -> List[str]:
    """``expr`` 이 읽는 외부 이름을 순서대로 반환한다.

    루프 변수, 빌트인, 그리고 expr 내부에서 바인딩되는 이름(컴프리헨션/제너레이터
    타깃, 람다 인자, 월러스 대상)은 제외한다.
    """
    bound = {loop_var}
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)

    names: List[str] = []
    seen = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name in bound or name in _BUILTIN_NAMES or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _build_parallel_call(
    for_node: ast.For,
    loop_var: str,
    target_name: str,
    expr: ast.expr,
    workers: Optional[ast.expr],
) -> ast.Assign:
    capture_names = _free_names(expr, loop_var)
    captures = ast.Dict(
        keys=[ast.Constant(value=n) for n in capture_names],
        values=[ast.Name(id=n, ctx=ast.Load()) for n in capture_names],
    )
    keywords = []
    if workers is not None:
        keywords.append(ast.keyword(arg="workers", value=workers))
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=_RT_ALIAS, ctx=ast.Load()),
            attr="parallel_map_into",
            ctx=ast.Load(),
        ),
        args=[
            ast.Name(id=target_name, ctx=ast.Load()),
            for_node.iter,
            ast.Constant(value=ast.unparse(expr)),
            ast.Constant(value=loop_var),
            captures,
        ],
        keywords=keywords,
    )
    return ast.Assign(
        targets=[ast.Name(id=target_name, ctx=ast.Store())], value=call
    )


def _process_body(stmts: List[ast.stmt], state: dict) -> List[ast.stmt]:
    new: List[ast.stmt] = []
    i = 0
    while i < len(stmts):
        node = stmts[i]
        _recurse(node, state)
        if _is_attr_marker(node):
            opts = _marker_options(node)
            nxt = stmts[i + 1] if i + 1 < len(stmts) else None
            if not isinstance(nxt, ast.For):
                raise UnsafeParallelLoop(
                    "attr[...] 마커 다음에는 for 루프가 와야 합니다."
                )
            _recurse(nxt, state)
            want_parallel = _is_truthy_const(opts.get("parallel"))
            want_unroll = _is_truthy_const(opts.get("unroll"))
            if want_parallel and want_unroll:
                raise UnrollError(
                    "attr 에 parallel 과 unroll 을 동시에 지정할 수 없습니다."
                )
            if want_parallel:
                shape = _independent_map(nxt)
                if shape is None:
                    raise UnsafeParallelLoop(
                        "병렬화하려면 'target[i] = expr' 형태의 독립 반복 map "
                        "이어야 하고, expr 안에서 target 을 읽지 않아야 합니다."
                    )
                loop_var, target_name, expr = shape
                new.append(
                    _build_parallel_call(
                        nxt, loop_var, target_name, expr, opts.get("workers")
                    )
                )
                state["used"] = True
            elif want_unroll:
                # 이후 unroll 패스가 처리하도록 노드에 플래그만 단다.
                setattr(nxt, _UNROLL_FLAG, True)
                new.append(nxt)
            else:
                # 알려진 키가 없으면 마커만 제거하고 루프는 그대로 둔다.
                new.append(nxt)
            i += 2
            continue
        new.append(node)
        i += 1
    return new


def _recurse(node: ast.AST, state: dict) -> None:
    for field in _BODY_FIELDS:
        value = getattr(node, field, None)
        if isinstance(value, list) and all(
            isinstance(s, ast.stmt) for s in value
        ):
            setattr(node, field, _process_body(value, state))
    for handler in getattr(node, "handlers", []):
        _recurse(handler, state)


def parallelize(tree: ast.Module) -> ast.Module:
    state = {"used": False}
    tree.body = _process_body(tree.body, state)
    if state["used"]:
        tree.body.insert(
            0,
            ast.Import(names=[ast.alias(name="pydya.runtime", asname=_RT_ALIAS)]),
        )
    ast.fix_missing_locations(tree)
    return tree
