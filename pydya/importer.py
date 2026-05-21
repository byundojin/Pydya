"""(실험적) import 시점 특수화 훅.

``from __future__ import annotations`` 처럼, 각 모듈이 상단에서 ::

    from pydya.module import specialize_here

마커를 import 하는 것으로 *스스로* opt-in 한다. 훅이 설치되면, import 되는
모듈의 소스에 이 마커가 있을 때만 :func:`compile_source` 로 부분 평가한 뒤
실행한다. 마커가 없는 모듈은 일절 건드리지 않는다.

사용::

    import pydya.importer as importer
    importer.install({'SCALE': 4})   # 컴파일 타임 값 공급 + 훅 설치
    import kernel                     # kernel 상단에 마커가 있으면 특수화됨

주의: ``from __future__`` 가 컴파일러에 의해 모듈 로드 *이전* 에 처리되는
것처럼, 이 훅도 대상 모듈이 처음 import 되기 *전에* 설치되어 있어야 한다.
이미 ``sys.modules`` 에 적재된 모듈은 다시 특수화되지 않는다.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from typing import Any, Dict, Mapping

from pydya.compiler import compile_source
from pydya.module import MARKER

_ENV: Dict[str, Any] = {}


class _Loader(importlib.abc.Loader):
    def __init__(self, origin: str, source: str):
        self.origin = origin
        self.source = source

    def create_module(self, spec):  # 기본 모듈 생성을 사용한다.
        return None

    def exec_module(self, module) -> None:
        compiled = compile_source(self.source, _ENV)
        code = compile(compiled, self.origin, "exec")
        exec(code, module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # pydya 자체 모듈은 특수화 대상에서 제외하여 재귀를 막는다.
        if fullname == "pydya" or fullname.startswith("pydya."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        try:
            with open(spec.origin, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return None
        if MARKER not in source:
            return None  # opt-in 마커가 없으면 기본 로더에 위임한다.
        return importlib.util.spec_from_file_location(
            fullname, spec.origin, loader=_Loader(spec.origin, source)
        )


def install(env: Mapping[str, Any] | None = None) -> None:
    """컴파일 타임 환경을 등록하고 import 훅을 설치한다."""
    if env:
        _ENV.update(env)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def reset() -> None:
    """등록된 환경을 비우고 import 훅을 제거한다(주로 테스트용)."""
    _ENV.clear()
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
