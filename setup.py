"""Pydya 의 C 확장 빌드 진입점.

pyproject.toml 이 메타데이터를, 이 파일이 ``ext_modules`` 만 담는다.
``python setup.py build_ext --inplace`` 또는 ``pip install -e .`` 로 빌드한다.
"""

from setuptools import Extension, setup

extensions = [
    Extension(
        name="pydya._tensor",
        sources=["pydya/_tensor.c"],
        # -O3 + -march=native 로 컴파일러 auto-vectorization 을 활용하고,
        # restrict 포인터·정렬된 버퍼와 결합해 SIMD 코드가 깔리도록 한다.
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-funroll-loops",
            "-Wall",
            "-Wextra",
        ],
    ),
]

setup(ext_modules=extensions)
