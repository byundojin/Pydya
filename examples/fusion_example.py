"""표현식 융합 (Phase 3) 예시.

``: Tensor`` 어노테이션이 붙은 함수의 ``a * b + c`` 패턴이 컴파일 타임에
단일 융합 호출 ``pydya._tensor.madd(a, b, c)`` 로 치환된다. 미융합 시 발생하던
임시 텐서 2개의 할당과 메모리 추가 순회가 사라진다.

실행:  PYTHONPATH=. python examples/fusion_example.py
"""

from pydya import Tensor, compile_source

SOURCE = """\
def fma(a: Tensor, b: Tensor, c: Tensor):
    return a * b + c

# 어노테이션 없으면 손대지 않는다 — Phase 2 의 개별 연산자로 실행.
def unfused(a, b, c):
    return a * b + c
"""


if __name__ == "__main__":
    compiled = compile_source(SOURCE)
    print("=== compiled ===")
    print(compiled)

    print("=== running ===")
    ns = {}
    exec(compiled, ns)
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0])
    c = Tensor([100.0, 200.0, 300.0, 400.0])
    print("fma(a, b, c)     =", ns["fma"](a, b, c).to_list())
    print("unfused(a, b, c) =", ns["unfused"](a, b, c).to_list())
