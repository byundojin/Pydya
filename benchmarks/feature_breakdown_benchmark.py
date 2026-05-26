"""기능별 기여도 측정 — MLP 추론을 4단계로 분해.

같은 forward pass (input → hidden → relu → output) 를 네 단계로 실행하고
*각 단계의 추가 기능* 이 얼마나 기여하는지 정량화한다.

    A) Pure Python      — list-of-lists matmul + relu, 인터프리터 그대로
    B) C scalar         — pydya.Tensor 의 ops, auto-vectorize 끈 변종 사용
    C) C vectorized     — pydya.Tensor 의 ops, -O3 -march=native 자동 SIMD
    D) C vec + fused    — compile_source 가 linear_relu 융합 호출로 lowering

기여도:
    A → B :  C 레벨 Tensor (Python 인터프리터 → 핫 C 루프)
    B → C :  Vector 최적화 (auto-vectorize / SIMD)
    C → D :  표현식 융합 (relu(W@x+b) → linear_relu)

attr[unroll] 은 *부분평가 substrate* 이며 위 추론 파이프라인의 핫 코드는
모두 C 안이라 runtime 기여 0 (별도 섹션에서 짧게 확인).

사이즈:
    small  : 64  → 32   → 10   (sanity, 모든 단계 빠름)
    medium : 256 → 128  → 10   (분 단위 pure Python)
    large  : 784 → 1024 → 10   (~5분 pure Python on 1000 추론)

실행:
    PYTHONPATH=. python benchmarks/feature_breakdown_benchmark.py [size]
    # size: small | medium | large | all  (기본 all)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

from pydya import Tensor, compile_source
from pydya._tensor import (
    add_scalar,
    linear_relu,
    matmul,
    matmul_scalar,
    relu,
    relu_scalar,
)

random.seed(0)

MODELS = {
    "small":  {"layers": [64,  32,         10], "n_samples": 1000},
    "medium": {"layers": [256, 128,        10], "n_samples": 1000},
    "large":  {"layers": [784, 1024,       10], "n_samples": 1000},
    # 3-layer huge — pure Python 으로 ~5분 이상 도는 워크로드
    "huge":   {"layers": [784, 2048, 1024, 10], "n_samples": 3000},
}

FORWARD_SRC_2LAYER = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""

FORWARD_SRC_3LAYER = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor, W3: Tensor, b3: Tensor):
    h1 = relu(W1 @ x + b1)
    h2 = relu(W2 @ h1 + b2)
    return W3 @ h2 + b3
"""


# ─── Stage A: Pure Python (list-of-lists, 인터프리터 그대로) ────────────


def _matvec_relu_py(W, b, x, apply_relu):
    rows = len(W)
    cols = len(x)
    out = [0.0] * rows
    for i in range(rows):
        acc = b[i]
        row = W[i]
        for j in range(cols):
            acc += row[j] * x[j]
        out[i] = (acc if acc > 0.0 else 0.0) if apply_relu else acc
    return out


def forward_pure_python(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = _matvec_relu_py(Ws[i], bs[i], h, apply_relu=(i < last))
    return h


# ─── Stage B: C Tensor scalar (auto-vec 끈 변종) ────────────────────────


def forward_c_scalar(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = add_scalar(matmul_scalar(Ws[i], h), bs[i])
        if i < last:
            h = relu_scalar(h)
    return h


# ─── Stage C: C Tensor vectorized (현재 빌드, -O3 -march=native) ────────


def forward_c_vec(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = matmul(Ws[i], h) + bs[i]
        if i < last:
            h = relu(h)
    return h


# ─── Stage D: C vec + fused (compile_source 로 lowering) ────────────────


def make_compiled_forward(n_layers):
    src = FORWARD_SRC_2LAYER if n_layers == 2 else FORWARD_SRC_3LAYER
    compiled = compile_source(src)
    ns = {}
    exec(compiled, ns)
    return ns["forward"], compiled


def call_compiled(forward_fn, x, Ws, bs):
    if len(Ws) == 2:
        return forward_fn(x, Ws[0], bs[0], Ws[1], bs[1])
    return forward_fn(x, Ws[0], bs[0], Ws[1], bs[1], Ws[2], bs[2])


# ─── 데이터 준비 ────────────────────────────────────────────────────────


def make_weights_py(layers):
    """layers = [in_dim, h1, h2, ..., out_dim] → 인접한 쌍마다 W, b 생성."""
    Ws, bs = [], []
    for i in range(len(layers) - 1):
        in_, out_ = layers[i], layers[i + 1]
        scale = (1.0 / in_) ** 0.5
        Ws.append([[random.gauss(0, scale) for _ in range(in_)] for _ in range(out_)])
        bs.append([random.gauss(0, 0.01) for _ in range(out_)])
    return Ws, bs


def make_samples_py(n, dim):
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


# ─── 측정 ──────────────────────────────────────────────────────────────


def time_stage_generic(forward, xs, Ws, bs):
    start = time.perf_counter()
    for x in xs:
        forward(x, Ws, bs)
    return time.perf_counter() - start


def time_stage_compiled(forward, xs, Ws, bs):
    start = time.perf_counter()
    for x in xs:
        call_compiled(forward, x, Ws, bs)
    return time.perf_counter() - start


def max_abs_diff(a, b):
    if hasattr(a, "to_list"): a = a.to_list()
    if hasattr(b, "to_list"): b = b.to_list()
    return max(abs(x - y) for x, y in zip(a, b))


def run_model(name, spec, do_pure_python=True):
    layers = spec["layers"]
    n = spec["n_samples"]
    arch_str = " → ".join(str(d) for d in layers)
    print(f"\n{'=' * 78}")
    print(f" model: {name}  ({arch_str}),  N = {n:,}")
    print('=' * 78)

    Ws_py, bs_py = make_weights_py(layers)
    xs_py = make_samples_py(n, layers[0])

    Ws_t = [Tensor(W) for W in Ws_py]
    bs_t = [Tensor(b) for b in bs_py]
    xs_t = [Tensor(x) for x in xs_py]

    forward_compiled, compiled_src = make_compiled_forward(len(Ws_py))
    if name == "small":
        print("compiled source (small 예시):")
        for line in compiled_src.strip().splitlines():
            print(f"    {line}")

    # 정합성 한 번
    ref_py = forward_pure_python(xs_py[0], Ws_py, bs_py)
    ref_sc = forward_c_scalar(xs_t[0], Ws_t, bs_t).to_list()
    ref_vc = forward_c_vec(xs_t[0], Ws_t, bs_t).to_list()
    ref_fd = call_compiled(forward_compiled, xs_t[0], Ws_t, bs_t).to_list()
    diff_sc = max_abs_diff(ref_py, ref_sc)
    diff_vc = max_abs_diff(ref_py, ref_vc)
    diff_fd = max_abs_diff(ref_py, ref_fd)

    # warm-up
    forward_c_scalar(xs_t[0], Ws_t, bs_t)
    forward_c_vec(xs_t[0], Ws_t, bs_t)
    call_compiled(forward_compiled, xs_t[0], Ws_t, bs_t)

    results = []
    if do_pure_python:
        t = time_stage_generic(forward_pure_python, xs_py, Ws_py, bs_py)
        results.append(("A) Pure Python      ", t, 0.0))
    t = time_stage_generic(forward_c_scalar, xs_t, Ws_t, bs_t)
    results.append(("B) C scalar (no-SIMD)", t, diff_sc))
    t = time_stage_generic(forward_c_vec, xs_t, Ws_t, bs_t)
    results.append(("C) C vectorized     ", t, diff_vc))
    t = time_stage_compiled(forward_compiled, xs_t, Ws_t, bs_t)
    results.append(("D) C vec + fused    ", t, diff_fd))

    base = results[0][1]
    print(f"\n{'단계':<24}{'총시간':>12}{'  per-inf':>14}{'  vs A':>10}{'  vs prev':>10}   diff vs A")
    print("-" * 78)
    prev = None
    for label, t, diff in results:
        per_inf = t / n
        ratio_a = base / t if t > 0 else float("inf")
        ratio_p = prev / t if prev else 1.0
        print(f"{label:<24}{t:>9.3f}s   {per_inf * 1e6:>9.2f} us   {ratio_a:>6.1f}x   {ratio_p:>6.2f}x   {diff:.2e}")
        prev = t

    print()
    if len(results) >= 4:
        ta, tb, tc, td = results[0][1], results[1][1], results[2][1], results[3][1]
        print("  단계 gap 별 기여 (시간 절감):")
        print(f"    C 레벨 Tensor    (A→B) : {ta:.2f}s → {tb:.3f}s   {(1 - tb / ta) * 100:>5.2f}% 단축")
        print(f"    Vector 최적화    (B→C) : {tb:.3f}s → {tc:.3f}s   {(1 - tc / tb) * 100:>5.2f}% 단축")
        print(f"    표현식 융합      (C→D) : {tc:.3f}s → {td:.3f}s   {(1 - td / tc) * 100:>5.2f}% 단축")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("=" * 78)
    print(" Pydya 기능별 기여도 벤치마크")
    print("=" * 78)
    print(f"  python : {sys.version.split()[0]}")
    print(f"  cpus   : {os.cpu_count()}")
    print(f"  target : {target}")

    if target == "all":
        # huge 는 pure Python 이 ~5분+ — 다른 size 다 끝난 뒤 마지막
        sizes = ["small", "medium", "large", "huge"]
    elif target in MODELS:
        sizes = [target]
    else:
        print(f"  invalid target: {target}")
        return

    for size in sizes:
        spec = MODELS[size]
        run_model(size, spec, do_pure_python=True)

    print(f"\n{'=' * 78}")
    print(" attr[unroll] 에 대한 정직한 한 줄:")
    print("   - 위 추론 파이프라인의 핫 코드는 모두 C 안이라 attr[unroll] 기여는 0.")
    print("   - attr[unroll] 의 가치는 부분평가 substrate (residual 코드 가독성,")
    print("     이후 fusion 패턴 매칭의 발판) 이며 runtime 가속이 아님.")
    print('=' * 78)


if __name__ == "__main__":
    main()
