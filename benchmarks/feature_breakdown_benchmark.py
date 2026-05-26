"""종합 추론 KPI 벤치마크 — 정밀판.

이 벤치는 KPI 다. *어떤 op 를 몇 번 돌렸는지 / min·max·mean 분포 / 왜 그
숫자가 나오는지* 까지 다 보인다.

측정 원칙:
  - 모든 측정은 _stats.time_stable 로 adaptive inner repeat + warmup. CoV < 10%
    목표. 표본수 × inner × 총 호출수 를 명시해 *무엇을 몇 번 돌렸는지* 한눈에.
  - 연산별 단독 측정 — 사전 계산된 입력으로 *해당 op 만* 격리해 호출. matmul
    / add / relu / linear_relu 각각 분포.
  - 단계별 forward 측정 — 같은 알고리즘을 4단계 (Pure Python / C scalar /
    C vectorized / Fused) 로 측정.
  - 이론 합 vs 실측 sanity — op 단독 합 ≈ stage forward 측정인지 점검.
  - 각 수치의 이론적 근거 (FLOPs, ns/op, 메모리 traffic) 함께 출력.

실행: PYTHONPATH=. python benchmarks/feature_breakdown_benchmark.py [size]
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

from pydya import Tensor, compile_source
from pydya._tensor import (
    add_scalar,
    linear_relu,
    matmul,
    matmul_scalar,
    relu,
    relu_scalar,
)

sys.path.insert(0, os.path.dirname(__file__))
from _stats import Stats, fmt_time

random.seed(0)

MODELS = {
    "small":  {"layers": [64,  32,         10]},
    "medium": {"layers": [256, 128,        10]},
    "large":  {"layers": [784, 1024,       10]},
    "huge":   {"layers": [784, 2048, 1024, 10]},
}

FORWARD_SRC_2L = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""
FORWARD_SRC_3L = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor,
            W3: Tensor, b3: Tensor):
    h1 = relu(W1 @ x + b1)
    h2 = relu(W2 @ h1 + b2)
    return W3 @ h2 + b3
"""

N_SAMPLES = 200       # 표본 수 (각 측정마다)
TARGET_OP_US = 100    # op 단독 측정 한 표본의 목표 시간
TARGET_STAGE_US = 500 # stage forward 측정 한 표본의 목표 시간
WARMUP = 10


def time_stable(fn: Callable, target_us: float, n_samples: int = N_SAMPLES):
    """안정 측정 — adaptive inner repeat 으로 *표본당 시간* 이 target 이상.

    반환: (Stats, inner, total_calls). total_calls = n_samples * inner 가
    *총 몇 번 호출했는지*.
    """
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-9)
    inner = max(1, int(target_us * 1e-6 / one))

    samples = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return Stats(samples), inner, n_samples * inner


# ─── 데이터 ───────────────────────────────────────────────────────────


def make_weights_py(layers):
    Ws, bs = [], []
    for i in range(len(layers) - 1):
        in_, out_ = layers[i], layers[i + 1]
        scale = (1.0 / in_) ** 0.5
        Ws.append([[random.gauss(0, scale) for _ in range(in_)] for _ in range(out_)])
        bs.append([random.gauss(0, 0.01) for _ in range(out_)])
    return Ws, bs


# ─── 단계별 forward ────────────────────────────────────────────────────


def _matvec_relu_py(W, b, x, do_relu):
    rows, cols = len(W), len(x)
    out = [0.0] * rows
    for i in range(rows):
        acc = b[i]
        row = W[i]
        for j in range(cols):
            acc += row[j] * x[j]
        out[i] = (acc if acc > 0.0 else 0.0) if do_relu else acc
    return out


def forward_pure_python(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = _matvec_relu_py(Ws[i], bs[i], h, do_relu=(i < last))
    return h


def forward_c_scalar(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = add_scalar(matmul_scalar(Ws[i], h), bs[i])
        if i < last:
            h = relu_scalar(h)
    return h


def forward_c_vec(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = matmul(Ws[i], h) + bs[i]
        if i < last:
            h = relu(h)
    return h


def make_compiled(n_layers):
    src = FORWARD_SRC_2L if n_layers == 2 else FORWARD_SRC_3L
    ns = {}
    exec(compile_source(src), ns)
    return ns["forward"]


def call_compiled(fn, x, Ws, bs):
    if len(Ws) == 2:
        return fn(x, Ws[0], bs[0], Ws[1], bs[1])
    return fn(x, Ws[0], bs[0], Ws[1], bs[1], Ws[2], bs[2])


# ─── 표 출력 ──────────────────────────────────────────────────────────


def print_row(label, n_samples, inner, total, s: Stats):
    print(
        f"  {label:<28} {n_samples:>5}×{inner:<6} ={total:>7,}  "
        f"{fmt_time(s.min):>9} {fmt_time(s.p50):>9} {fmt_time(s.mean):>9} "
        f"{fmt_time(s.p99):>9} {fmt_time(s.std):>9} {s.cov*100:>5.1f}%"
    )


def print_header():
    print(
        f"  {'label':<28} {'samples×inner':<13} {'총호출':>9}  "
        f"{'min':>9} {'p50':>9} {'mean':>9} {'p99':>9} {'std':>9} {'CoV':>6}"
    )
    print("  " + "─" * 110)


# ─── 메인 ─────────────────────────────────────────────────────────────


def run_model(name, spec):
    layers = spec["layers"]
    n_layers = len(layers) - 1
    arch = " → ".join(str(d) for d in layers)

    print(f"\n{'═' * 116}")
    print(f" model: {name}  ({arch})")
    print('═' * 116)

    Ws_py, bs_py = make_weights_py(layers)
    Ws_t = [Tensor(W) for W in Ws_py]
    bs_t = [Tensor(b) for b in bs_py]
    x_py = [random.gauss(0, 1) for _ in range(layers[0])]
    x_t = Tensor(x_py)

    forward_compiled = make_compiled(n_layers)

    # 사전 계산: 각 op 격리 측정에 필요한 중간 상태
    h_inputs = [x_t]  # h_inputs[li] = layer li 의 입력 (이미 relu 까지 통과)
    for li in range(n_layers):
        mm = matmul(Ws_t[li], h_inputs[li])
        s = mm + bs_t[li]
        if li < n_layers - 1:
            h_inputs.append(relu(s))

    # ── 연산별 단독 측정 ───────────────────────────────────────────
    print("\n## (1) 연산별 단독 측정 — 사전 계산된 입력으로 그 op 만 격리 반복")
    print_header()

    op_means: Dict[str, float] = {}

    for li in range(n_layers):
        in_ = layers[li]
        out_ = layers[li + 1]
        x_in = h_inputs[li]
        W_in = Ws_t[li]
        b_in = bs_t[li]
        mm_pre = matmul(W_in, x_in)
        sum_pre = mm_pre + b_in

        # matmul
        s, inner, total = time_stable(lambda: matmul(W_in, x_in), TARGET_OP_US)
        print_row(f"matmul L{li+1} ({out_}×{in_})", N_SAMPLES, inner, total, s)
        op_means[f"matmul_L{li+1}"] = s.mean

        # add (tensor + tensor)
        s, inner, total = time_stable(lambda: mm_pre + b_in, TARGET_OP_US)
        print_row(f"add L{li+1} ({out_},)", N_SAMPLES, inner, total, s)
        op_means[f"add_L{li+1}"] = s.mean

        if li < n_layers - 1:
            # relu
            s, inner, total = time_stable(lambda: relu(sum_pre), TARGET_OP_US)
            print_row(f"relu L{li+1} ({out_},)", N_SAMPLES, inner, total, s)
            op_means[f"relu_L{li+1}"] = s.mean

            # linear_relu (융합 커널)
            s, inner, total = time_stable(
                lambda: linear_relu(W_in, x_in, b_in), TARGET_OP_US
            )
            print_row(f"linear_relu L{li+1}", N_SAMPLES, inner, total, s)
            op_means[f"linear_relu_L{li+1}"] = s.mean

    # ── 단계별 forward 측정 ───────────────────────────────────────
    print("\n## (2) 단계별 forward 측정 — 같은 입력으로 forward 전체 반복")
    print_header()

    stage_means: Dict[str, float] = {}

    s, inner, total = time_stable(
        lambda: forward_pure_python(x_py, Ws_py, bs_py), TARGET_STAGE_US
    )
    print_row("A) Pure Python (list)", N_SAMPLES, inner, total, s)
    stage_means["A"] = s.mean

    s, inner, total = time_stable(
        lambda: forward_c_scalar(x_t, Ws_t, bs_t), TARGET_STAGE_US
    )
    print_row("B) C scalar (no-SIMD)", N_SAMPLES, inner, total, s)
    stage_means["B"] = s.mean

    s, inner, total = time_stable(
        lambda: forward_c_vec(x_t, Ws_t, bs_t), TARGET_STAGE_US
    )
    print_row("C) C vectorized", N_SAMPLES, inner, total, s)
    stage_means["C"] = s.mean

    s, inner, total = time_stable(
        lambda: call_compiled(forward_compiled, x_t, Ws_t, bs_t), TARGET_STAGE_US
    )
    print_row("D) C vec + fused", N_SAMPLES, inner, total, s)
    stage_means["D"] = s.mean

    # ── 이론 합 vs 실측 sanity ────────────────────────────────────
    print("\n## (3) 이론 합 vs 실측 — op 단독 합 ≈ forward stage?")
    unfused_sum = 0.0
    for li in range(n_layers):
        unfused_sum += op_means[f"matmul_L{li+1}"]
        unfused_sum += op_means[f"add_L{li+1}"]
        if li < n_layers - 1:
            unfused_sum += op_means[f"relu_L{li+1}"]
    fused_sum = 0.0
    for li in range(n_layers - 1):
        fused_sum += op_means[f"linear_relu_L{li+1}"]
    fused_sum += op_means[f"matmul_L{n_layers}"]
    fused_sum += op_means[f"add_L{n_layers}"]

    diff_unfused = abs(stage_means["C"] - unfused_sum) / stage_means["C"] * 100
    diff_fused = abs(stage_means["D"] - fused_sum) / stage_means["D"] * 100
    print(f"  unfused 이론 합 (op 단독 mean 합)  = {fmt_time(unfused_sum)}")
    print(f"  unfused 실측 (C vectorized stage) = {fmt_time(stage_means['C'])}   "
          f"(diff {diff_unfused:.1f}%)")
    print(f"  fused 이론 합 (op 단독 mean 합)    = {fmt_time(fused_sum)}")
    print(f"  fused 실측 (D Fused stage)         = {fmt_time(stage_means['D'])}   "
          f"(diff {diff_fused:.1f}%)")

    # ── stage gap 분석 ────────────────────────────────────────────
    print("\n## (4) 단계 gap (mean 기준)")
    a, b, c, d = stage_means["A"], stage_means["B"], stage_means["C"], stage_means["D"]
    print(f"  A → B  C 레벨 핫루프         : {fmt_time(a)} → {fmt_time(b)}   "
          f"{a/b:>5.1f}x   단축 {(1-b/a)*100:>5.2f}%")
    print(f"  B → C  auto-vectorize/SIMD   : {fmt_time(b)} → {fmt_time(c)}   "
          f"{b/c:>5.2f}x   단축 {(1-c/b)*100:>5.2f}%")
    print(f"  C → D  linear_relu 융합      : {fmt_time(c)} → {fmt_time(d)}   "
          f"{c/d:>5.2f}x   단축 {(1-d/c)*100:>5.2f}%")

    # ── 이론적 근거 출력 ──────────────────────────────────────────
    total_mac = sum(layers[i] * layers[i + 1] for i in range(n_layers))
    print(f"\n## (5) 측정 근거 (왜 이 수치인가)")
    print(f"  모델 컴퓨트:")
    print(f"    matmul 총 {total_mac:,} multiply-add")
    print(f"    bias add {sum(layers[i+1] for i in range(n_layers))} elem,  "
          f"relu {sum(layers[i+1] for i in range(n_layers-1))} elem")
    print(f"")
    print(f"  Pure Python (A): {fmt_time(a)}")
    py_ns_per_op = a / total_mac * 1e9
    print(f"    {total_mac:,} ops 에 {fmt_time(a)} → {py_ns_per_op:.1f}ns/op.")
    print(f"    CPython 산술 op 평균 ~30-50ns/op 와 일치 (인터프리터 BINARY_OP "
          f"+ BINARY_SUBSCR + STORE).")
    print(f"")
    print(f"  C 핫루프 (B/C/D): C vec={fmt_time(c)}, fused={fmt_time(d)}")
    c_ns_per_op = c / total_mac * 1e9
    print(f"    C vec: {c_ns_per_op:.2f}ns/op.  CPU 1 cycle ≈ 0.3ns 기준 "
          f"~{c_ns_per_op/0.3:.1f} cycle/op")
    if c_ns_per_op / 0.3 < 1.0:
        print(f"    < 1 cycle/op → 명백히 SIMD 활성. AVX2 (8 floats/cycle) 또는 ILP")
    elif c_ns_per_op / 0.3 < 3.0:
        print(f"    1-3 cycle/op → scalar-near-peak 또는 memory-bound.")
    else:
        print(f"    > 3 cycle/op → memory-bound 또는 호출 오버헤드 비중 큼.")
    print(f"")
    simd_ratio = b / c
    print(f"  SIMD 효과 (B→C): {b/c:.2f}x")
    print(f"    이론 최대 AVX2 = 8x. 실측 {simd_ratio:.2f}x 인 이유 = ")
    print(f"    (1) scalar baseline 도 -O3 -funroll-loops 라 매우 빠름")
    print(f"    (2) 매트멀 크기 클수록 memory-bound 로 천장 도달")
    print(f"")
    fusion_saving = c - d
    print(f"  Fusion 절감 (C→D): {fmt_time(fusion_saving)}")
    if n_layers >= 1 and f"add_L1" in op_means and f"relu_L1" in op_means:
        add_relu_sum = op_means["add_L1"] + op_means["relu_L1"]
        print(f"    이론 출처 = (add L1 + relu L1) 호출/alloc 오버헤드 "
              f"= {fmt_time(add_relu_sum)} 의 일부 절감.")
        print(f"    fused linear_relu 가 한 inner loop 안에 mul+add+relu 모두 처리해")
        print(f"    임시 텐서 2개 alloc 와 메모리 2회 추가 패스 제거.")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("═" * 116)
    print(" Pydya 종합 추론 KPI 벤치마크 — 정밀판")
    print(" 각 측정: 표본 200개 × adaptive inner (sample 당 ≥ {0}us op / {1}us stage)".format(
        TARGET_OP_US, TARGET_STAGE_US))
    print("═" * 116)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print(f"  target : {target}")

    if target == "all":
        sizes = ["small", "medium", "large", "huge"]
    elif target in MODELS:
        sizes = [target]
    else:
        print(f"  invalid target: {target}")
        return

    for size in sizes:
        run_model(size, MODELS[size])


if __name__ == "__main__":
    main()
