"""``CompileVar`` 선언을 수집하고 컴파일 타임 전용 노드를 제거한다."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Mapping, Tuple


class MissingCompileValue(KeyError):
    """선언된 CompileVar 의 값이 환경에 없을 때 발생한다."""


def _is_compilevar_func(func: ast.expr) -> bool:
    """호출 대상이 ``CompileVar`` 또는 ``CompileVar[...]`` 인지 확인한다."""
    if isinstance(func, ast.Subscript):
        return _is_compilevar_func(func.value)
    return (isinstance(func, ast.Name) and func.id == "CompileVar") or (
        isinstance(func, ast.Attribute) and func.attr == "CompileVar"
    )


def _compilevar_decl(node: ast.stmt) -> Tuple[str, str] | None:
    """CompileVar 선언 대입이면 ``(변수이름, 레이블)`` 을, 아니면 None.

    지원 형태::

        V = CompileVar[int]()   # 레이블 = 변수 이름 'V'
        V = CompileVar()        # 레이블 = 변수 이름 'V'
        V = CompileVar('flag')  # 레이블 = 명시한 'flag'
    """
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
    ):
        return None
    call = node.value
    if not _is_compilevar_func(call.func):
        return None
    var_name = node.targets[0].id
    if (
        call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        label = call.args[0].value
    else:
        label = var_name
    return var_name, label


def _is_pydya_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        return mod == "pydya" or mod.startswith("pydya.")
    if isinstance(node, ast.Import):
        return any(
            alias.name == "pydya" or alias.name.startswith("pydya.")
            for alias in node.names
        )
    return False


def collect_in_body(
    body: List[ast.stmt], env: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[ast.stmt]]:
    """문장 리스트에서 CompileVar 선언/ pydya import 를 제거한다.

    ``(정적값_매핑, 정리된_문장리스트)`` 를 반환한다. 모듈 본문이든 함수
    본문이든 동일하게 적용할 수 있다(데코레이터/임포트 경로에서 재사용).
    """
    static_values: Dict[str, Any] = {}
    new_body: List[ast.stmt] = []
    for node in body:
        if _is_pydya_import(node):
            continue
        decl = _compilevar_decl(node)
        if decl is not None:
            var_name, label = decl
            if label not in env:
                raise MissingCompileValue(
                    f"CompileVar({label!r}) has no value in env"
                )
            static_values[var_name] = env[label]
            continue
        new_body.append(node)
    return static_values, new_body


def collect_static_env(
    tree: ast.Module, env: Mapping[str, Any]
) -> Dict[str, Any]:
    """모듈 트리에서 CompileVar 선언과 pydya import 를 제거하고 정적 환경 반환.

    반환되는 매핑의 키는 소스에서 바인딩된 *변수 이름* 이며
    (예: ``V = CompileVar[int]()`` 의 ``V``), 값은 CompileVar 레이블로
    ``env`` 에서 가져온 구체적인 값이다.
    """
    static_values, tree.body = collect_in_body(tree.body, env)
    return static_values
