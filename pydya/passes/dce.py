"""부작용 없는 대입에 대한 죽은 저장(dead store) 제거.

``X`` 가 모듈 어디에서도 로드되지 않으면 ``X = <순수 식>`` 을 제거한다.
부작용 없는 우변만 대상이 되므로 문장을 제거해도 관찰 가능한 동작은
바뀌지 않는다. 죽은 저장 하나를 제거하면 앞선 저장이 죽을 수 있으므로
고정점(fixpoint)에 도달할 때까지 반복한다.

모듈 최상위 바인딩은 절대 제거하지 않는다. 다른 모듈에서 import 될 수
있어 이 소스만으로는 미사용임을 증명할 수 없기 때문이다. 따라서 제거는
함수 본문으로 한정한다.
"""

from __future__ import annotations

import ast

_PURE_EXPR_NODES = (
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
)


def _is_pure(expr: ast.expr) -> bool:
    for node in ast.walk(expr):
        if not isinstance(node, (_PURE_EXPR_NODES + (ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop))):
            return False
    return True


def _used_names(tree: ast.AST) -> set:
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _dead_target(node: ast.stmt, used: set) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return False
    return target.id not in used and _is_pure(node.value)


def _prune_bodies(node: ast.AST, used: set, prunable: bool) -> bool:
    """``node`` 아래의 문장 리스트에서 죽은 대입을 제거한다.

    모듈 최상위 본문을 순회하는 동안에는 ``prunable`` 이 False 이므로
    export 가능성이 있는 항목을 보존하고, 함수 내부로 들어가면 True 가 된다.
    """
    inside_func = prunable or isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    )
    changed = False
    for field, value in ast.iter_fields(node):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            if inside_func:
                kept = [s for s in value if not _dead_target(s, used)]
                if len(kept) != len(value):
                    setattr(node, field, kept)
                    changed = True
            else:
                kept = value
            for stmt in kept:
                changed |= _prune_bodies(stmt, used, inside_func)
        elif isinstance(value, ast.AST):
            changed |= _prune_bodies(value, used, inside_func)
    return changed


def eliminate_dead_code(tree: ast.AST) -> ast.AST:
    while True:
        used = _used_names(tree)
        if not _prune_bodies(tree, used, prunable=False):
            break
    ast.fix_missing_locations(tree)
    return tree
