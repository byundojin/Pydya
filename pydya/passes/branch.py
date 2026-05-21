"""정적으로 결정된 조건에 대한 분기 제거.

폴딩 이후에 실행되므로 정적 값에만 의존하는 조건은 이미 상수로 접혀 있다.
이런 분기는 실제로 선택되는 쪽만 남기고 잘라낸다.
"""

from __future__ import annotations

import ast
from typing import List, Union


class _BranchPruner(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> Union[ast.AST, List[ast.stmt]]:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            # 선택된 본문(비어 있을 수도 있음)을 그 자리에 끼워 넣는다.
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_While(self, node: ast.While) -> Union[ast.AST, List[ast.stmt]]:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and not node.test.value:
            return node.orelse
        return node


def eliminate_branches(tree: ast.AST) -> ast.AST:
    pruned = _BranchPruner().visit(tree)
    ast.fix_missing_locations(pruned)
    return pruned
