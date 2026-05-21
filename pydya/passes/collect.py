"""Collect ``CompileVar`` declarations and strip compile-time-only nodes."""

from __future__ import annotations

import ast
from typing import Any, Dict, Mapping


class MissingCompileValue(KeyError):
    """Raised when a declared CompileVar has no value in the environment."""


def _compilevar_label(call: ast.Call) -> str | None:
    """Return the string label of a ``CompileVar('label')`` call, else None."""
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
    """Strip CompileVar declarations / pydya imports and return the static env.

    The returned mapping is keyed by the *variable name* bound in the source
    (e.g. ``V`` in ``V = CompileVar('V')``) and holds the concrete value drawn
    from ``env`` under the CompileVar label.
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
