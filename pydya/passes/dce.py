"""Dead store elimination for side-effect-free assignments.

Removes ``X = <pure expr>`` when ``X`` is never loaded anywhere in the
module. Only side-effect-free right-hand sides are eligible, so removing the
statement cannot change observable behaviour. Runs to a fixpoint because
removing one dead store can make an earlier one dead.

Module top-level bindings are never removed: they may be imported by other
modules, so we cannot prove them unused from this source alone. Pruning is
therefore limited to function bodies.
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
    """Remove dead assignments from statement lists under ``node``.

    ``prunable`` is False while walking the module's own top-level body so
    that potential exports are preserved; it becomes True inside functions.
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
