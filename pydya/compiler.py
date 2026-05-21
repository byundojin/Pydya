"""Python 소스를 부분 평가된 소스로 변환하는 패스 파이프라인."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Optional

from pydya.passes.branch import eliminate_branches
from pydya.passes.collect import collect_static_env
from pydya.passes.dce import eliminate_dead_code
from pydya.passes.fold import fold
from pydya.passes.inline import inline_calls


def compile_source(source: str, env: Optional[Mapping[str, Any]] = None) -> str:
    """컴파일 타임 환경 ``env`` 를 기준으로 ``source`` 를 부분 평가한다.

    ``env`` 는 ``CompileVar(...)`` 에 전달한 이름을 컴파일 타임 값으로
    매핑한다. 변환된 소스를 문자열로 반환한다.
    """
    env = dict(env or {})
    tree = ast.parse(source)
    static_values = collect_static_env(tree, env)
    fold(tree, static_values)
    eliminate_branches(tree)
    inline_calls(tree)
    eliminate_dead_code(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
