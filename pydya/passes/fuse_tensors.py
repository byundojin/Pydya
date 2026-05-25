"""Tensor 표현식 트리 융합 패스.

함수 인자 어노테이션 ``: Tensor`` 를 *신뢰* (Pydya 의 CompileVar/attr 와 같은
결의 trust-based 모델) 해 어떤 Name 이 Tensor 인지 파악하고, 함수 본문 안에서
tensor-producing 식의 결과로 바인딩되는 지역변수도 같이 추적해 다음 패턴을
단일 융합 호출로 치환한다.

지원 패턴:

* ``a * b + c``                     → ``__pydya_t.madd(a, b, c)``
* ``c + a * b``                      → ``__pydya_t.madd(a, b, c)``
* ``relu(W @ x + b)``                → ``__pydya_t.linear_relu(W, x, b)``
* ``relu(matmul(W, x) + b)``         → ``__pydya_t.linear_relu(W, x, b)``
* ``relu(b + W @ x)``                → ``__pydya_t.linear_relu(W, x, b)``

또한 융합과 별개로, 함수 본문에 남아 있는 bare ``relu`` / ``matmul`` /
``madd`` / ``linear_relu`` Name 호출을 ``__pydya_t.<name>`` 으로 자동 qualify
한다 (사용자가 ``from pydya._tensor import ...`` 안 해도 동작하도록).

지원 안 함: 스칼라 섞임, Subscript 피연산자, 깊은 체인의 자동 분해, mul-sub.
"""

from __future__ import annotations

import ast
from typing import Set

_T_ALIAS = "__pydya_t"
_TENSOR_FUNCS = frozenset({"relu", "matmul", "madd", "linear_relu"})


def _is_tensor_annotation(node: ast.expr | None) -> bool:
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


def _is_tensor_func_name(node: ast.expr) -> bool:
    """relu/matmul/madd/linear_relu 호출인지 (bare 또는 __pydya_t.<name>)."""
    if isinstance(node, ast.Name) and node.id in _TENSOR_FUNCS:
        return True
    if isinstance(node, ast.Attribute) and node.attr in _TENSOR_FUNCS:
        return True
    return False


def _expr_is_tensor(node: ast.expr, tensors: Set[str]) -> bool:
    """``node`` 가 Tensor 를 생성하는 식인지 (보수적 추론)."""
    if isinstance(node, ast.Name):
        return node.id in tensors
    if isinstance(node, ast.Call):
        if _is_tensor_func_name(node.func):
            return True
        return False
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.MatMult)
    ):
        return _expr_is_tensor(node.left, tensors) and _expr_is_tensor(
            node.right, tensors
        )
    return False


def _propagate_tensor_locals(
    func: ast.FunctionDef | ast.AsyncFunctionDef, seed: Set[str]
) -> Set[str]:
    """단순 ``name = <tensor 식>`` 대입의 LHS 를 Tensor 집합에 누적한다.

    함수 본문 statements 를 순차 1회 통과 — Python 의 일반 실행 순서와 동일.
    중첩 함수는 분리된 스코프이므로 내려가지 않는다.
    """
    tensors = set(seed)
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _expr_is_tensor(stmt.value, tensors)
        ):
            tensors.add(stmt.targets[0].id)
    return tensors


class _Fuser(ast.NodeTransformer):
    def __init__(self, tensors: Set[str]):
        self.tensors = tensors
        self.changed = False
        self.uses_tensor_funcs = False  # bare relu/matmul/... 등장 여부

    def _is_tensor_name(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in self.tensors
        )

    def _extract_matmul(self, node: ast.AST):
        """node 가 W@x (BinOp MatMult) 또는 matmul(W, x) (Call, bare 또는
        ``__pydya_t.matmul``) 이면 (W, x) 반환."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return node.left, node.right
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 2
            and not node.keywords
        ):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "matmul") or (
                isinstance(f, ast.Attribute) and f.attr == "matmul"
            ):
                return node.args[0], node.args[1]
        return None

    def _is_relu_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
            return False
        f = node.func
        if isinstance(f, ast.Name) and f.id == "relu":
            return True
        if isinstance(f, ast.Attribute) and f.attr == "relu":
            return True
        return False

    def visit_Call(self, node: ast.Call):  # noqa: N802
        self.generic_visit(node)
        # relu(...) 패턴 매칭 시도
        if self._is_relu_call(node):
            inner = node.args[0]
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
                # left + right 양쪽 매칭 시도
                for matmul_side, bias_side in (
                    (inner.left, inner.right),
                    (inner.right, inner.left),
                ):
                    wx = self._extract_matmul(matmul_side)
                    if wx is None:
                        continue
                    W, x = wx
                    if (
                        self._is_tensor_name(W)
                        and self._is_tensor_name(x)
                        and self._is_tensor_name(bias_side)
                    ):
                        self.changed = True
                        return self._build_call("linear_relu", [W, x, bias_side], node)
        # bare tensor 함수명 → __pydya_t.<name> 으로 자동 qualify
        if isinstance(node.func, ast.Name) and node.func.id in _TENSOR_FUNCS:
            self.uses_tensor_funcs = True
            node.func = ast.Attribute(
                value=ast.Name(id=_T_ALIAS, ctx=ast.Load()),
                attr=node.func.id,
                ctx=ast.Load(),
            )
        return node

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        self.generic_visit(node)
        # MADD 패턴: (a*b) + c / c + (a*b)
        if isinstance(node.op, ast.Add):
            left, right = node.left, node.right
            if (
                isinstance(left, ast.BinOp)
                and isinstance(left.op, ast.Mult)
                and self._is_tensor_name(left.left)
                and self._is_tensor_name(left.right)
                and self._is_tensor_name(right)
            ):
                self.changed = True
                return self._build_call("madd", [left.left, left.right, right], node)
            if (
                isinstance(right, ast.BinOp)
                and isinstance(right.op, ast.Mult)
                and self._is_tensor_name(right.left)
                and self._is_tensor_name(right.right)
                and self._is_tensor_name(left)
            ):
                self.changed = True
                return self._build_call("madd", [right.left, right.right, left], node)
        return node

    def _build_call(self, name: str, args, ref: ast.AST) -> ast.Call:
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_T_ALIAS, ctx=ast.Load()),
                attr=name,
                ctx=ast.Load(),
            ),
            args=list(args),
            keywords=[],
        )
        return ast.copy_location(call, ref)


def _strip_tensor_annotations(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    """``: Tensor`` 어노테이션을 제거 (def 시점에 Tensor 이름 해석 불필요)."""
    args = func.args
    for arg_list in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in arg_list:
            if _is_tensor_annotation(a.annotation):
                a.annotation = None
    if _is_tensor_annotation(func.returns):
        func.returns = None


def _has_pydya_t_import(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return True
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "pydya._tensor" and alias.asname == _T_ALIAS:
                    return True
    return False


def fuse_tensors(tree: ast.AST) -> ast.AST:
    """모든 함수 정의 안의 텐서 표현식 패턴을 융합 호출로 lowering 한다."""
    used = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seed = _collect_tensor_args(node)
            if seed:
                # 본문 내 지역변수도 단계적으로 Tensor 로 인정
                tensors = _propagate_tensor_locals(node, seed)
                fuser = _Fuser(tensors)
                for stmt in node.body:
                    fuser.visit(stmt)
                if fuser.changed or fuser.uses_tensor_funcs:
                    used = True
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
