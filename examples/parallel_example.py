"""Nadya 스타일 ``attr[{...}]`` 병렬 for 루프를 보여준다.

실행:  PYTHONPATH=. python examples/parallel_example.py
"""

from pydya import compile_source

SOURCE = """\
from pydya import attr

out = [0] * 8
factor = 10

attr[{'parallel': True, 'workers': 4}]
for i in range(8):
    out[i] = i * i + factor

print(out)
"""

if __name__ == "__main__":
    compiled = compile_source(SOURCE)
    print("=== compiled ===")
    print(compiled)
    print("=== running ===")
    exec(compiled, {})
