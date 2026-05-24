"""XOR 추론 — Pydya 컴파일러 위에서 도는 최소 신경망.

가중치 하드코딩 (학습 아님). ``compile_source`` 가 두 개의 ``relu(W @ x + b)``
패턴을 단일 융합 호출 ``__pydya_t.linear_relu`` 로 lowering 한다.

실행:  PYTHONPATH=. python examples/xor_inference.py
"""

from pydya import Tensor, compile_source

SOURCE = """\
def xor_mlp(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return relu(W2 @ h + b2)
"""


def main():
    compiled = compile_source(SOURCE)
    print("=== compiled ===")
    print(compiled)
    print()

    ns = {}
    exec(compiled, ns)
    xor_mlp = ns["xor_mlp"]

    # XOR 을 푸는 고전적 가중치 (relu 활성화 기준)
    W1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = Tensor([0.0, -1.0])
    W2 = Tensor([[1.0, -2.0]])
    b2 = Tensor([0.0])

    print("=== XOR 진리표 ===")
    print("  x1  x2 | expected | got")
    print("  ─── ─── ┼ ──────── ┼ ────")
    for x1 in (0.0, 1.0):
        for x2 in (0.0, 1.0):
            expected = float(int(x1) ^ int(x2))
            x = Tensor([x1, x2])
            y = xor_mlp(x, W1, b1, W2, b2).to_list()[0]
            mark = "OK " if y == expected else "FAIL"
            print(f"  {x1!s:>3} {x2!s:>3} | {expected!s:>8} | {y!s:>3}  {mark}")


if __name__ == "__main__":
    main()
