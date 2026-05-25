"""사전학습 손글씨 숫자 MLP 추론 — 컴파일 전 vs 컴파일 후 부분별 비교.

같은 forward pass 를 두 가지로 실행:

* **PRE-COMPILE**  — 원본 소스 그대로. ``relu(W @ x + b)`` 가 그때그때
  ``Tensor.__matmul__`` / ``Tensor.__add__`` / 모듈 ``relu`` 를 차례로
  호출 → 임시 텐서 3개 + 메모리 3회 추가 순회.
* **COMPILED**    — ``compile_source`` 가 fuse_tensors 패스로 lowering 한
  결과. ``linear_relu(W1, x, b1)`` 단일 호출 + 단일 메모리 순회.

레이어별로 분해 측정해서 *어느 부분에서 어느 정도 좋아지는지* 까지 보인다.
실행:  PYTHONPATH=. python benchmarks/digit_inference_benchmark.py
"""

import json
import os
import sys
import time
from pathlib import Path

from pydya import Tensor, compile_source
from pydya._tensor import linear_relu, matmul, relu

WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "digit_weights.json"
)
ITERS = 20_000  # 64-d 입력이라 한 forward 가 매우 빠름 — 안정 측정용

FORWARD_SRC = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""


# ─── PRE-COMPILE 경로: 사용자가 짠 그대로의 Python ──────────────────────


def forward_precompile(x, W1, b1, W2, b2):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2


def hidden_precompile(x, W1, b1):
    return relu(W1 @ x + b1)


def output_precompile(h, W2, b2):
    return W2 @ h + b2


# ─── COMPILED 경로: compile_source 가 lowering 한 결과 ──────────────────


def make_compiled():
    src = compile_source(FORWARD_SRC)
    ns = {}
    exec(src, ns)
    return ns["forward"], src


def hidden_compiled(x, W1, b1):
    return linear_relu(W1, x, b1)


def output_compiled(h, W2, b2):
    # compile_source 의 출력 레이어 (relu 없음) 는 융합 대상 아님 — pre 와 동일.
    return matmul(W2, h) + b2


# ─── 측정 헬퍼 ─────────────────────────────────────────────────────────


def bench(fn, args, iters=ITERS):
    fn(*args)  # warm-up
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    return (time.perf_counter() - start) / iters, out  # 평균 per-iter


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a.to_list(), b.to_list()))


# ─── 메인 ──────────────────────────────────────────────────────────────


def main():
    payload = json.loads(WEIGHTS_PATH.read_text())
    w = payload["weights"]
    W1 = Tensor(w["W1"])
    b1 = Tensor(w["b1"])
    W2 = Tensor(w["W2"])
    b2 = Tensor(w["b2"])
    samples = payload["samples"]
    x = Tensor(samples["x"][0])  # 한 샘플로 per-iter 시간 측정
    arch = payload["architecture"]

    forward_compiled, compiled_src = make_compiled()

    print("=" * 72)
    print(" 손글씨 숫자 MLP 추론 — PRE-COMPILE vs COMPILED")
    print("=" * 72)
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  cpus     : {os.cpu_count()}")
    print(f"  arch     : {arch['input']} -> {arch['hidden']} -> {arch['output']} (relu hidden)")
    print(f"  iters    : {ITERS:,}")
    print("-" * 72)
    print("  compiled source:")
    for line in compiled_src.strip().splitlines():
        print(f"      {line}")
    print("-" * 72)

    # 1) Hidden layer 단독 — 융합이 적용되는 곳
    t_hp, r_hp = bench(hidden_precompile, (x, W1, b1))
    t_hc, r_hc = bench(hidden_compiled, (x, W1, b1))
    diff_h = max_abs_diff(r_hp, r_hc)

    # 2) Output layer 단독 — 융합 대상 아님 (relu 없음)
    h = relu(W1 @ x + b1)
    t_op, r_op = bench(output_precompile, (h, W2, b2))
    t_oc, r_oc = bench(output_compiled, (h, W2, b2))
    diff_o = max_abs_diff(r_op, r_oc)

    # 3) 전체 forward
    t_fp, r_fp = bench(forward_precompile, (x, W1, b1, W2, b2))
    t_fc, r_fc = bench(forward_compiled, (x, W1, b1, W2, b2))
    diff_f = max_abs_diff(r_fp, r_fc)

    def line(label, t_pre, t_post, diff, fused):
        speedup = t_pre / t_post if t_post > 0 else float("inf")
        marker = "★" if fused else " "
        print(
            f"  {marker} {label:<32} "
            f"{t_pre * 1e6:>8.2f} us  →  {t_post * 1e6:>8.2f} us  "
            f"= {speedup:5.2f}x   (diff {diff:g})"
        )

    print("  레이어                              PRE-COMPILE        COMPILED            speedup")
    print("  " + "-" * 70)
    line("hidden  (relu(W1 @ x + b1))", t_hp, t_hc, diff_h, fused=True)
    line("output  (W2 @ h + b2)", t_op, t_oc, diff_o, fused=False)
    line("FULL forward", t_fp, t_fc, diff_f, fused=True)

    # 4) 50 샘플 실제 워크로드
    xs = [Tensor(v) for v in samples["x"]]

    def run_all_pre():
        return [forward_precompile(xv, W1, b1, W2, b2) for xv in xs]

    def run_all_compiled():
        return [forward_compiled(xv, W1, b1, W2, b2) for xv in xs]

    # 워밍업
    run_all_pre()
    run_all_compiled()

    REPEAT = 200
    start = time.perf_counter()
    for _ in range(REPEAT):
        run_all_pre()
    t_all_pre = (time.perf_counter() - start) / REPEAT
    start = time.perf_counter()
    for _ in range(REPEAT):
        run_all_compiled()
    t_all_post = (time.perf_counter() - start) / REPEAT
    n = len(xs)
    print("-" * 72)
    print(f"  50 샘플 전체 1회   PRE={t_all_pre * 1e3:.2f} ms  COMPILED={t_all_post * 1e3:.2f} ms"
          f"  speedup {t_all_pre / t_all_post:.2f}x")
    print(f"  샘플당 평균       PRE={t_all_pre / n * 1e6:.2f} us  COMPILED={t_all_post / n * 1e6:.2f} us")
    print("=" * 72)
    print(" ★ = fuse_tensors 패스가 융합을 적용한 부분")


if __name__ == "__main__":
    main()
