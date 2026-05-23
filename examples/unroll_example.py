"""Nadya 스타일 컴파일 타임 unroll 을 보여준다.

``W = CompileVar('W')`` 로 컴파일 타임에 고정되는 반복 횟수를 두면,
``for i in range(W)`` 가 ``i = 0..W-1`` 로 완전히 펼쳐진다. Nadya 의
``template<>`` 메타프로그래밍이 SIMD 슬롯폭만큼 펼치는 것의 Python 재현.

실행:  PYTHONPATH=. python examples/unroll_example.py
"""

from pydya import compile_source

SOURCE = """\
W = CompileVar('W')

def dot_product(a, b):
    result = 0
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
