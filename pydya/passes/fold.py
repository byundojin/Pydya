"""정적(CompileVar) 값에 대한 상수 전파 및 폴딩.

CompileVar 값에 바인딩된 이름만 치환한다. 일반 바인딩은 우변이 상수더라도
런타임 잔여 코드로 그대로 남긴다.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, Mapping, Type

_BINOPS: Dict[Type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}

_UNARYOPS: Dict[Type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}

_CMPOPS: Dict[Type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _const(value: Any, ref: ast.AST) -> ast.Constant:
    return ast.copy_location(ast.Constant(value), ref)


class _Folder(ast.NodeTransformer):
    def __init__(self, static_values: Mapping[str, Any]):
        self.static = static_values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.static:
            return _const(self.static[node.id], node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        fn = _BINOPS.get(type(node.op))
        if fn and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                return _const(fn(node.left.value, node.right.value), node)
            except Exception:
                return node
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        fn = _UNARYOPS.get(type(node.op))
        if fn and isinstance(node.operand, ast.Constant):
            try:
                return _const(fn(node.operand.value), node)
            except Exception:
                return node
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if not all(isinstance(v, ast.Constant) for v in node.values):
            return node
        values = [v.value for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for v in values:
                result = v
                if not v:
                    break
        else:  # Or 연산
            result = False
            for v in values:
                result = v
                if v:
                    break
        return _const(result, node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        operands = [node.left, *node.comparators]
        if not all(isinstance(o, ast.Constant) for o in operands):
            return node
        if not all(type(op) in _CMPOPS for op in node.ops):
            return node
        try:
            result = True
            for op, left, right in zip(node.ops, operands, operands[1:]):
                result = _CMPOPS[type(op)](left.value, right.value)
                if not result:
                    break
        except Exception:
            return node
        return _const(result, node)


def fold(tree: ast.AST, static_values: Mapping[str, Any]) -> ast.AST:
    """정적 이름을 치환하고 그 결과로 생긴 상수 식을 폴딩한다."""
    folded = _Folder(static_values).visit(tree)
    ast.fix_missing_locations(folded)
    return folded
