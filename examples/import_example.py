"""import 시점 특수화(opt-in)를 보여준다.

실행:  PYTHONPATH=. python examples/import_example.py
"""

import os
import sys

import pydya.importer as importer

# 대상 모듈(_kernel)을 import 경로에 노출한다.
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    importer.configure({"SCALE": 4}, modules={"_kernel"})
    import _kernel

    print("=== _kernel.scaled(10) with SCALE = 4 ===")
    print(_kernel.scaled(10))  # 40
    print("SCALE folded away:", not hasattr(_kernel, "SCALE"))
