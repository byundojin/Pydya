"""Branch elimination for statically decided conditions.

Runs after folding, so a condition that depends only on static values has
already collapsed to a constant. Such branches are pruned to the taken side.
"""

from __future__ import annotations

import ast
from typing import List, Union


class _BranchPruner(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> Union[ast.AST, List[ast.stmt]]:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            # Returning the taken suite (possibly empty) splices it in place.
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
