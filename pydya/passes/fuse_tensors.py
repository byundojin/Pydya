"""Tensor 표현식 트리 융합 패스 (Phase 3).

함수 인자 어노테이션 ``: Tensor`` 를 *신뢰* 해 (Pydya 의 CompileVar/attr 와
같은 결의 trust-based 모델) 어떤 Name 이 Tensor 인지 파악하고, ``a * b + c``
같은 FMA 패턴을 단일 융합 커널 호출 ``pydya._tensor.madd(a, b, c)`` 로 치환한다.

지원 패턴 (1차):

* ``a * b + c``  (모두 Tensor Name) → ``madd(a, b, c)``
* ``c + a * b``  (덧셈 commutative)  → ``madd(a, b, c)``

치환 효과: Phase 2 의 미융합 식 ``a * b + c`` 는 임시 텐서 2개를 만들고
메모리를 3회 추가 순회한다(약 24N bytes traffic). 융합 호출은 단일 할당
단일 순회(약 16N bytes). 메모리-바운드 영역에서 추가 가속이 따라온다.

미지원 (확장 여지):

* Subscript/Attribute 피연산자 (e.g., ``a[0] * b[0] + c[0]``)
* ``a * b - c`` 같은 mul-sub
* 깊은 체인 (``a*b*c + d``)
* 어노테이션이 없는 함수, 어노테이션이 string 이 아닌 다른 형태(`Optional[Tensor]` 등)

위 미지원 케이스는 손대지 않으며, Phase 2 의 개별 연산자 호출이 그대로 실행된다.
"""

from __future__ import annotations

import ast
from typing import Set

_T_ALIAS = "__pydya_t"


def _is_tensor_annotation(node: ast.expr | None) -> bool:
    """``: Tensor`` 또는 ``: 'Tensor'`` 어노테이션인지."""
    if isinstance(node, ast.Name) and node.id == "Tensor":
        return True
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == "Tensor"
    ):
        return True
    return False


def _collect_tensor_args(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Set[str]:
    names: Set[str] = set()
    args = func.args
    for arg_list in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in arg_list:
            if _is_tensor_annotation(a.annotation):
                names.add(a.arg)
    return names


class _Fuser(ast.NodeTransformer):
    def __init__(self, tensors: Set[str]):
        self.tensors = tensors
        self.changed = False

    def _is_tensor_name(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in self.tensors
        )

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        # 안쪽 표현식부터 먼저 처리 (재귀 후 바깥 패턴 재검사 가능하도록).
        self.generic_visit(node)
        if not isinstance(node.op, ast.Add):
            return node
        left, right = node.left, node.right
        # 패턴 1: (a * b) + c
        if (
            isinstance(left, ast.BinOp)
            and isinstance(left.op, ast.Mult)
            and self._is_tensor_name(left.left)
            and self._is_tensor_name(left.right)
            and self._is_tensor_name(right)
        ):
            self.changed = True
            return self._build_madd(left.left, left.right, right, node)
        # 패턴 2: c + (a * b)
        if (
            isinstance(right, ast.BinOp)
            and isinstance(right.op, ast.Mult)
            and self._is_tensor_name(right.left)
            and self._is_tensor_name(right.right)
            and self._is_tensor_name(left)
        ):
            self.changed = True
            return self._build_madd(right.left, right.right, left, node)
        return node

    def _build_madd(
        self,
        a: ast.expr,
        b: ast.expr,
        c: ast.expr,
        ref: ast.AST,
    ) -> ast.Call:
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_T_ALIAS, ctx=ast.Load()),
                attr="madd",
                ctx=ast.Load(),
            ),
            args=[a, b, c],
            keywords=[],
        )
        return ast.copy_location(call, ref)


def _has_pydya_t_import(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return True  # 모듈이 아니면 건드리지 않음
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "pydya._tensor" and alias.asname == _T_ALIAS:
                    return True
    return False


def _strip_tensor_annotations(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    """함수의 ``: Tensor`` / ``-> Tensor`` 어노테이션을 제거한다.

    어노테이션 정보는 fuse_tensors 가 이미 추출했으므로, 컴파일 결과 코드가
    def 시점에 ``Tensor`` 이름을 해석할 필요를 없애기 위해 strip 한다(컴파일
    결과를 exec 할 때 ``Tensor`` import 가 없어도 NameError 가 나지 않게).
    """
    args = func.args
    for arg_list in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in arg_list:
            if _is_tensor_annotation(a.annotation):
                a.annotation = None
    if _is_tensor_annotation(func.returns):
        func.returns = None


def fuse_tensors(tree: ast.AST) -> ast.AST:
    """모든 함수 정의 안의 텐서 FMA 패턴을 융합 호출로 lowering 한다."""
    used = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tensors = _collect_tensor_args(node)
            if tensors:
                fuser = _Fuser(tensors)
                for stmt in node.body:
                    fuser.visit(stmt)
                if fuser.changed:
                    used = True
            # 어노테이션은 (융합 발화 여부와 무관하게) 정보를 다 뽑은 뒤
            # strip 해서 exec 시 Tensor 이름 의존을 제거한다.
            _strip_tensor_annotations(node)
    if used and not _has_pydya_t_import(tree):
        tree.body.insert(
            0,
            ast.Import(
                names=[ast.alias(name="pydya._tensor", asname=_T_ALIAS)]
            ),
        )
    ast.fix_missing_locations(tree)
    return tree
