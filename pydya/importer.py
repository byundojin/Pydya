"""(실험적) import 시점 특수화 훅.

명시적으로 등록한 모듈에 한해, import 될 때 소스를 :func:`compile_source` 로
부분 평가한 뒤 실행한다. 안전을 위해 allowlist 기반 opt-in 이며, 등록되지
않은 모듈은 일절 건드리지 않는다.

사용::

    import pydya.importer as importer
    importer.configure({'SCALE': 4}, modules={'kernel'})
    import kernel        # kernel 의 CompileVar 가 4 로 고정되어 로드된다

주의: 대상 모듈이 이미 import 되어 ``sys.modules`` 에 있으면 다시 특수화되지
않는다. ``configure`` 는 첫 import 이전에 호출해야 한다.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from typing import Any, Dict, Iterable, Mapping, Set

from pydya.compiler import compile_source

_ENV: Dict[str, Any] = {}
_MODULES: Set[str] = set()


class _Loader(importlib.abc.Loader):
    def __init__(self, origin: str):
        self.origin = origin

    def create_module(self, spec):  # 기본 모듈 생성을 사용한다.
        return None

    def exec_module(self, module) -> None:
        with open(self.origin, "r", encoding="utf-8") as f:
            source = f.read()
        compiled = compile_source(source, _ENV)
        code = compile(compiled, self.origin, "exec")
        exec(code, module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _MODULES:
            return None
        # 등록된 모듈의 실제 위치를 표준 경로 탐색으로 찾는다(자기 자신 제외).
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        return importlib.util.spec_from_file_location(
            fullname, spec.origin, loader=_Loader(spec.origin)
        )


def _install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def configure(
    env: Mapping[str, Any] | None = None,
    *,
    modules: Iterable[str] = (),
) -> None:
    """컴파일 타임 환경과 특수화 대상 모듈 allowlist 를 등록하고 훅을 설치한다."""
    if env:
        _ENV.update(env)
    _MODULES.update(modules)
    _install()


def reset() -> None:
    """등록된 환경/모듈을 비우고 import 훅을 제거한다(주로 테스트용)."""
    _ENV.clear()
    _MODULES.clear()
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
