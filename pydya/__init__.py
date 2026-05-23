"""Pydya: Python 소스를 위한 Nadya 스타일 부분 평가 컴파일러."""

from pydya.attr import attr
from pydya.compilevar import CompileVar
from pydya.compiler import compile_source
from pydya.importer import install
from pydya.specialize import specialize

try:
    # C 확장은 setup.py build_ext --inplace (또는 pip install -e .) 로 빌드된다.
    # 빌드 전이면 Tensor 는 import 되지 않지만, 컴파일러 파이프라인은 그대로 동작.
    from pydya._tensor import Tensor  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover
    Tensor = None  # type: ignore[assignment]

__all__ = [
    "CompileVar",
    "attr",
    "compile_source",
    "specialize",
    "install",
    "Tensor",
]
