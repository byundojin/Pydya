"""Python 소스를 부분 평가된 소스로 변환하는 패스 파이프라인."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Optional

from pydya.passes.branch import eliminate_branches
from pydya.passes.collect import collect_static_env
from pydya.passes.dce import eliminate_dead_code
from pydya.passes.fold import fold
from pydya.passes.inline import inline_calls
from pydya.passes.parallelize import parallelize
from pydya.passes.unroll import mark_template_loops, unroll


def optimize(tree: ast.AST, static_values: Mapping[str, Any]) -> ast.AST:
    """정적 값을 기준으로 단형화→폴딩→분기제거→인라인→DCE 파이프라인을 적용.

    모듈 트리든 함수 정의 노드든 동일하게 동작한다(데코레이터 경로에서 함수
    본문에 직접 적용하기 위해 공유한다).
    """
    # 마킹은 fold 이전(Name 'W' 가 아직 살아 있을 때).
    mark_template_loops(tree, static_values)
    fold(tree, static_values)
    unroll(tree)
    eliminate_branches(tree)
    inline_calls(tree)
    eliminate_dead_code(tree)
    ast.fix_missing_locations(tree)
    return tree


def compile_source(source: str, env: Optional[Mapping[str, Any]] = None) -> str:
    """컴파일 타임 환경 ``env`` 를 기준으로 ``source`` 를 부분 평가한다.

    ``env`` 는 ``CompileVar(...)`` 에 전달한 이름을 컴파일 타임 값으로
    매핑한다. 변환된 소스를 문자열로 반환한다.

    파이프라인 순서: collect → mark → fold → parallelize → unroll → branch
    → inline → dce. mark 는 CompileVar 출신 값에 의존하는 for 루프를 표시
    하고(이후 unroll 의 대상), parallelize 는 ``attr[parallel]`` 마커가 붙은
    for 를 먼저 병렬 호출로 lowering 한다(unroll 이 그 자리를 건드리지
    않도록).
    """
    env = dict(env or {})
    tree = ast.parse(source)
    static_values = collect_static_env(tree, env)
    mark_template_loops(tree, static_values)
    fold(tree, static_values)
    parallelize(tree)
    unroll(tree)
    eliminate_branches(tree)
    inline_calls(tree)
    eliminate_dead_code(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
