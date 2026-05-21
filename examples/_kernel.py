"""import 특수화 데모용 대상 모듈 (examples/import_example.py 에서 사용)."""

from pydya import CompileVar
from pydya.importer import specialize_here  # import 시점 특수화 opt-in

SCALE = CompileVar[int]()


def scaled(x):
    return x * SCALE
