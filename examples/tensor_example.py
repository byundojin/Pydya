"""C 레벨 Tensor primitive 의 기본 사용법.

빌드:  python setup.py build_ext --inplace
실행:  PYTHONPATH=. python examples/tensor_example.py
"""

from pydya import Tensor


def main():
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0])

    print("a       =", a)
    print("b       =", b)
    print("a * b   =", (a * b).to_list())  # element-wise
    print("a + b   =", (a + b).to_list())
    print("a * 2   =", (a * 2).to_list())  # 스칼라 broadcast
    print("len(a)  =", len(a))
    print("a[2]    =", a[2])

    a[0] = 99.0
    print("after a[0]=99.0:", a)

    # 크기 인자로 fill 초기화
    z = Tensor(8, fill=1.5)
    print("Tensor(8, fill=1.5):", z)


if __name__ == "__main__":
    main()
