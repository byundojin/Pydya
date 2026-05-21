"""import 시점 특수화(opt-in)를 보여준다.

실행:  PYTHONPATH=. python examples/import_example.py
"""

import os
import sys

import pydya.importer as importer

# 대상 모듈(_kernel)을 import 경로에 노출한다.
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    importer.install({"SCALE": 4})  # 값 공급 + 훅 설치 (import 이전에)
    import _kernel  # 상단의 specialize_here 마커로 opt-in 되어 특수화됨

    print("=== _kernel.scaled(10) with SCALE = 4 ===")
    print(_kernel.scaled(10))  # 40
    print("SCALE folded away:", not hasattr(_kernel, "SCALE"))
