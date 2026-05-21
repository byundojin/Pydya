"""Pass pipeline that turns Python source into partially evaluated source."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Optional

from pydya.passes.branch import eliminate_branches
from pydya.passes.collect import collect_static_env
from pydya.passes.fold import fold


def compile_source(source: str, env: Optional[Mapping[str, Any]] = None) -> str:
    """Partially evaluate ``source`` against the compile-time environment.

    ``env`` maps the names passed to ``CompileVar(...)`` to their compile-time
    values. Returns the transformed source as a string.
    """
    env = dict(env or {})
    tree = ast.parse(source)
    static_values = collect_static_env(tree, env)
    fold(tree, static_values)
    eliminate_branches(tree)
    # Further passes (inlining, DCE) are wired in here.
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
