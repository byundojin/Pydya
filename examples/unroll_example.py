"""Nadya 스타일 컴파일 타임 unroll 을 보여준다.

``attr[{'unroll': True}]`` 마커 한 줄로 다음 for 루프를 컴파일 타임에 펼친다.
``range`` 인자는 컴파일 타임에 상수로 결정되어야 한다(예: ``CompileVar``).

주의 — 이 unroll 은 현재 *부분평가의 substrate* 다. CPython 바이트코드 VM
위에서는 펼친 코드가 자동으로 빨라지지 않는다. 진짜 가속은 Phase 2 (C Tensor) /
Phase 3 (표현식 융합) 에서 본문이 네이티브로 lowering 된 뒤에 따라온다.

실행:  PYTHONPATH=. python examples/unroll_example.py
"""

from pydya import compile_source

SOURCE = """\
from pydya import attr
W = CompileVar('W')

def dot_product(a, b):
    result = 0
    attr[{'unroll': True}]
    for i in range(W):
        result += a[i] * b[i]
    return result

# 호출 결과까지 보이도록 즉시 실행
print(dot_product([1, 2, 3, 4], [10, 20, 30, 40]))
"""


if __name__ == "__main__":
    compiled = compile_source(SOURCE, env={"W": 4})
    print("=== compiled (W = 4) ===")
    print(compiled)
    print("=== running compiled output ===")
    exec(compiled, {})
