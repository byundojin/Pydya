"""Pass pipeline that turns Python source into partially evaluated source."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Optional


def compile_source(source: str, env: Optional[Mapping[str, Any]] = None) -> str:
    """Partially evaluate ``source`` against the compile-time environment.

    ``env`` maps the names passed to ``CompileVar(...)`` to their compile-time
    values. Returns the transformed source as a string.
    """
    env = dict(env or {})
    tree = ast.parse(source)
    # Passes are wired in here as they are implemented.
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
