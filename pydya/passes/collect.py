"""``CompileVar`` 선언을 수집하고 컴파일 타임 전용 노드를 제거한다."""

from __future__ import annotations

import ast
from typing import Any, Dict, Mapping


class MissingCompileValue(KeyError):
    """선언된 CompileVar 의 값이 환경에 없을 때 발생한다."""


def _compilevar_label(call: ast.Call) -> str | None:
    """``CompileVar('label')`` 호출의 문자열 레이블을 반환, 아니면 None."""
    func = call.func
    is_compilevar = (isinstance(func, ast.Name) and func.id == "CompileVar") or (
        isinstance(func, ast.Attribute) and func.attr == "CompileVar"
    )
    if not is_compilevar:
        return None
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant):
        return None
    label = call.args[0].value
    return label if isinstance(label, str) else None


def _is_pydya_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "pydya"
    if isinstance(node, ast.Import):
        return any(alias.name == "pydya" for alias in node.names)
    return False


def collect_static_env(
    tree: ast.Module, env: Mapping[str, Any]
) -> Dict[str, Any]:
    """CompileVar 선언과 pydya import 를 제거하고 정적 환경을 반환한다.

    반환되는 매핑의 키는 소스에서 바인딩된 *변수 이름* 이며
    (예: ``V = CompileVar('V')`` 의 ``V``), 값은 CompileVar 레이블로
    ``env`` 에서 가져온 구체적인 값이다.
    """
    static_values: Dict[str, Any] = {}
    new_body = []
    for node in tree.body:
        if _is_pydya_import(node):
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            label = _compilevar_label(node.value)
            if label is not None:
                if label not in env:
                    raise MissingCompileValue(
                        f"CompileVar({label!r}) has no value in env"
                    )
                static_values[node.targets[0].id] = env[label]
                continue
        new_body.append(node)
    tree.body = new_body
    return static_values
