"""``@specialize`` 데코레이터로 함수 단위 특수화를 보여준다.

실행:  PYTHONPATH=. python examples/decorator_example.py
"""

from pydya import CompileVar, specialize


@specialize({"V": 3})
def f(a):
    V = CompileVar[int]()
    if V < 5:
        return a + V
    else:
        return a * V


if __name__ == "__main__":
    print("=== specialized source (V = 3) ===")
    print(f.__pydya_source__)
    print("=== running ===")
    print(f(10))  # 13
