"""Nadya 스타일 ``attr[{...}]`` 병렬 for 루프를 병렬 실행 코드로 변환한다.

Nadya 의 ``attr[Parallel : true] for(...)`` 와 같은 모델이다. for 루프 바로
앞의 ``attr[{'parallel': True}]`` 마커를 인식하고, **반복 간 파괴적 갱신이
없는 독립 map** 인지 보수적으로 검사한 뒤 :func:`pydya.runtime.parallel_map_into`
호출로 lowering 한다. (Nadya 가 destructive update 가 없을 때만 자동
병렬화하는 것과 동일한 안전 규칙.)

1차 지원 형태 — 미리 할당된 리스트로의 독립 반복 map::

    attr[{'parallel': True}]
    for i in <iter>:
        out[i] = <expr>     # out 을 읽지 않고, 인덱스는 정확히 i

이외의 형태(누적 append, 외부 변수 갱신, 다중 문장 등)는 파괴적 갱신으로
보고 :class:`UnsafeParallelLoop` 를 던진다.
"""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple

_RT_ALIAS = "__pydya_rt"
_BODY_FIELDS = ("body", "orelse", "finalbody")


class UnsafeParallelLoop(Exception):
    """``attr`` 로 병렬을 요청했으나 안전한 독립 map 으로 증명할 수 없을 때."""


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


def _build_parallel_call(
    for_node: ast.For,
    loop_var: str,
    target_name: str,
    expr: ast.expr,
    workers: Optional[ast.expr],
) -> ast.Assign:
    lam = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=loop_var)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=expr,
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
            lam,
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
            if _is_truthy_const(opts.get("parallel")):
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
            else:
                # parallel 이 아니면 마커만 제거하고 직렬 루프로 둔다.
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
