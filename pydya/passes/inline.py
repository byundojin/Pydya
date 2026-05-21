"""상수 인자로 호출되는 단순 함수를 특수화하여 인라인한다.

"단순" 함수란 본문이 단일 ``return <expr>`` 이고 위치 인자만 받는 모듈
최상위 ``def`` 를 말한다. 이런 함수가 모두 상수인 인자로 호출되면, 호출은
파라미터를 치환·폴딩한 반환식으로 대체된다. 이는 Nadya 가 템플릿을
단형화(monomorphize)할 때 수행하는 베타 축약과 동일하다.

상수 인자일 때만 인라인하므로 런타임 식이 중복되거나 순서가 바뀌는 일은 없다.
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
