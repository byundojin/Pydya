"""Specialize and inline calls to simple functions with constant arguments.

A "simple" function is a module-level ``def`` whose body is a single
``return <expr>`` and which takes only positional parameters. When such a
function is called with all-constant arguments, the call is replaced by the
return expression with the parameters substituted and folded -- the same
beta-reduction Nadya performs when monomorphizing a template.

Only constant arguments are inlined, so no runtime expression is ever
duplicated or reordered.
"""

from __future__ import annotations

import ast
import copy
from typing import Dict, Tuple

from pydya.passes.fold import fold

SimpleFunc = Tuple[list, ast.expr]


def _collect_simple_funcs(tree: ast.Module) -> Dict[str, SimpleFunc]:
    funcs: Dict[str, SimpleFunc] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
            continue
        if args.defaults or args.kw_defaults:
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            continue
        ret = node.body[0].value
        if ret is None:
            continue
        funcs[node.name] = ([a.arg for a in args.args], ret)
    return funcs


class _Inliner(ast.NodeTransformer):
    def __init__(self, funcs: Dict[str, SimpleFunc]):
        self.funcs = funcs

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        spec = self.funcs.get(node.func.id)
        if spec is None:
            return node
        params, ret = spec
        if len(node.args) != len(params):
            return node
        if not all(isinstance(a, ast.Constant) for a in node.args):
            return node
        bindings = {p: a.value for p, a in zip(params, node.args)}
        body = fold(copy.deepcopy(ret), bindings)
        return ast.copy_location(body, node)


def inline_calls(tree: ast.Module) -> ast.AST:
    funcs = _collect_simple_funcs(tree)
    if not funcs:
        return tree
    inlined = _Inliner(funcs).visit(tree)
    ast.fix_missing_locations(inlined)
    return inlined
