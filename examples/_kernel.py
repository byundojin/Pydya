"""import 특수화 데모용 대상 모듈 (examples/import_example.py 에서 사용)."""

from pydya import CompileVar

SCALE = CompileVar[int]()


def scaled(x):
    return x * SCALE
