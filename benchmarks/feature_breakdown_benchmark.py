"""종합 추론 KPI 벤치마크 — 기능별 기여도 + 연산별 시간 분포.

세 가지 정직한 측정:

  1) **단계별 forward time** (분포까지)
       A   Pure Python (list-of-list)
       A'  Pure Python (array.array, 평탄 contiguous)
       B   C Tensor scalar (auto-vectorize 끈 변종)
       C   C Tensor vectorized
       D   C vec + 융합 (compile_source → linear_relu)

     gap 의 의미:
       A   → A'  데이터 레이아웃 (list-of-list → 평탄 contiguous, Python loop 유지)
       A'  → B   네이티브 루프 (Python loop → C 핫루프)
       B   → C   Vector 최적화 (auto-vectorize / SIMD)
       C   → D   표현식 융합 (relu(W@x+b) → linear_relu)

  2) **연산별 호출 통계** — forward 한 번에 어떤 연산이 몇 번 호출되고
     각각이 얼마나 걸리는지 (min/p50/mean/p99/std). 시간이 어디서 가는지
     투명하게.

  3) **모델 크기 sweep** — small/medium/large/huge. fusion 효과가 모델 크기에
     따라 어떻게 변하는지 가시화.

실행:  PYTHONPATH=. python benchmarks/feature_breakdown_benchmark.py [size]
"""

from __future__ import annotations

import array
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
from _stats import Stats, fmt_time, time_samples

random.seed(0)

MODELS = {
    "small":  {"layers": [64,  32,         10], "n_samples": 1000},
    "medium": {"layers": [256, 128,        10], "n_samples": 1000},
    "large":  {"layers": [784, 1024,       10], "n_samples": 1000},
    "huge":   {"layers": [784, 2048, 1024, 10], "n_samples": 1000},
}

FORWARD_SRC_2 = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)
    return W2 @ h + b2
"""

FORWARD_SRC_3 = """\
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor, W3: Tensor, b3: Tensor):
    h1 = relu(W1 @ x + b1)
    h2 = relu(W2 @ h1 + b2)
    return W3 @ h2 + b3
"""


# ─── 데이터 (5가지 표현) ───────────────────────────────────────────────


def make_weights_py(layers):
    """List-of-list 형태 (A 단계용)."""
    Ws, bs = [], []
    for i in range(len(layers) - 1):
        in_, out_ = layers[i], layers[i + 1]
        scale = (1.0 / in_) ** 0.5
        Ws.append([[random.gauss(0, scale) for _ in range(in_)] for _ in range(out_)])
        bs.append([random.gauss(0, 0.01) for _ in range(out_)])
    return Ws, bs


def make_weights_array(weights_py):
    """같은 가중치를 array.array('f') 평탄 contiguous 로 (A' 단계용)."""
    Ws_arr = []
    bs_arr = []
    Ws_py, bs_py = weights_py
    for W, b in zip(Ws_py, bs_py):
        rows, cols = len(W), len(W[0])
        flat = array.array("f", [0.0] * (rows * cols))
        for i in range(rows):
            for j in range(cols):
                flat[i * cols + j] = W[i][j]
        Ws_arr.append((flat, rows, cols))
        bs_arr.append(array.array("f", b))
    return Ws_arr, bs_arr


def make_weights_tensor(weights_py):
    """같은 가중치를 pydya.Tensor 로 (B/C/D 단계용)."""
    Ws_py, bs_py = weights_py
    return [Tensor(W) for W in Ws_py], [Tensor(b) for b in bs_py]


def make_samples_py(n, dim):
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


# ─── Stage A: Pure Python list-of-list ─────────────────────────────────


def _matvec_relu_listlist(W, b, x, apply_relu):
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


def forward_listlist(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = _matvec_relu_listlist(Ws[i], bs[i], h, apply_relu=(i < last))
    return h


# ─── Stage A': Pure Python with array.array (flat) ─────────────────────


def _matvec_relu_array(W_tuple, b, x, apply_relu):
    W_flat, rows, cols = W_tuple
    out = array.array("f", [0.0] * rows)
    for i in range(rows):
        acc = b[i]
        base = i * cols
        for j in range(cols):
            acc += W_flat[base + j] * x[j]
        out[i] = (acc if acc > 0.0 else 0.0) if apply_relu else acc
    return out


def forward_array(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = _matvec_relu_array(Ws[i], bs[i], h, apply_relu=(i < last))
    # array.array → list 로 한 번 변환 (정합성 비교용)
    return list(h)


# ─── Stage B: C scalar ─────────────────────────────────────────────────


def forward_c_scalar(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = add_scalar(matmul_scalar(Ws[i], h), bs[i])
        if i < last:
            h = relu_scalar(h)
    return h


# ─── Stage C: C vectorized ────────────────────────────────────────────


def forward_c_vec(x, Ws, bs):
    h = x
    last = len(Ws) - 1
    for i in range(len(Ws)):
        h = matmul(Ws[i], h) + bs[i]
        if i < last:
            h = relu(h)
    return h


# ─── Stage D: compile_source → fused ──────────────────────────────────


def make_compiled_forward(n_layers):
    src = FORWARD_SRC_2 if n_layers == 2 else FORWARD_SRC_3
    compiled = compile_source(src)
    ns = {}
    exec(compiled, ns)
    return ns["forward"], compiled


def call_compiled(fn, x, Ws, bs):
    if len(Ws) == 2:
        return fn(x, Ws[0], bs[0], Ws[1], bs[1])
    return fn(x, Ws[0], bs[0], Ws[1], bs[1], Ws[2], bs[2])


# ─── 측정 ──────────────────────────────────────────────────────────────


def time_one_pass(forward_fn, xs, *args) -> List[float]:
    """xs 전체에 대해 forward 한 번씩, 각 호출의 시간을 리스트로 반환."""
    times = []
    for x in xs:
        start = time.perf_counter()
        forward_fn(x, *args)
        times.append(time.perf_counter() - start)
    return times


def time_one_pass_compiled(fn, xs, Ws, bs) -> List[float]:
    times = []
    for x in xs:
        start = time.perf_counter()
        call_compiled(fn, x, Ws, bs)
        times.append(time.perf_counter() - start)
    return times


def profile_per_op_c_vec(xs_t, Ws_t, bs_t) -> Dict[str, List[float]]:
    """C vec 경로의 각 연산을 forward 안에서 개별 측정. 미융합 기준."""
    n_layers = len(Ws_t)
    op_times: Dict[str, List[float]] = {f"matmul L{i+1}": [] for i in range(n_layers)}
    op_times.update({f"add L{i+1}": [] for i in range(n_layers)})
    for i in range(n_layers - 1):
        op_times[f"relu L{i+1}"] = []

    for x in xs_t:
        h = x
        for li in range(n_layers):
            t0 = time.perf_counter()
            mm = matmul(Ws_t[li], h)
            t1 = time.perf_counter()
            ab = mm + bs_t[li]
            t2 = time.perf_counter()
            op_times[f"matmul L{li+1}"].append(t1 - t0)
            op_times[f"add L{li+1}"].append(t2 - t1)
            if li < n_layers - 1:
                t3 = time.perf_counter()
                h = relu(ab)
                t4 = time.perf_counter()
                op_times[f"relu L{li+1}"].append(t4 - t3)
            else:
                h = ab
    return op_times


def profile_per_op_fused(xs_t, Ws_t, bs_t) -> Dict[str, List[float]]:
    """융합 경로의 각 연산 — linear_relu 가 matmul+add+relu 를 한 호출로 묶음."""
    n_layers = len(Ws_t)
    op_times: Dict[str, List[float]] = {}
    for i in range(n_layers - 1):
        op_times[f"linear_relu L{i+1}"] = []
    op_times[f"matmul L{n_layers}"] = []
    op_times[f"add L{n_layers}"] = []

    for x in xs_t:
        h = x
        for li in range(n_layers - 1):
            t0 = time.perf_counter()
            h = linear_relu(Ws_t[li], h, bs_t[li])
            t1 = time.perf_counter()
            op_times[f"linear_relu L{li+1}"].append(t1 - t0)
        t0 = time.perf_counter()
        mm = matmul(Ws_t[-1], h)
        t1 = time.perf_counter()
        ab = mm + bs_t[-1]
        t2 = time.perf_counter()
        op_times[f"matmul L{n_layers}"].append(t1 - t0)
        op_times[f"add L{n_layers}"].append(t2 - t1)
    return op_times


def max_abs_diff(a, b):
    if hasattr(a, "to_list"): a = a.to_list()
    if hasattr(b, "to_list"): b = b.to_list()
    return max(abs(x - y) for x, y in zip(a, b))


def run_model(name, spec, skip_pure_python=False):
    layers = spec["layers"]
    n = spec["n_samples"]
    arch_str = " → ".join(str(d) for d in layers)
    print(f"\n{'━' * 92}")
    print(f" model: {name}  ({arch_str}),  N samples = {n:,}")
    print('━' * 92)

    Ws_py, bs_py = make_weights_py(layers)
    Ws_arr, bs_arr = make_weights_array((Ws_py, bs_py))
    Ws_t, bs_t = make_weights_tensor((Ws_py, bs_py))
    xs_py = make_samples_py(n, layers[0])
    xs_arr = [array.array("f", x) for x in xs_py]
    xs_t = [Tensor(x) for x in xs_py]

    forward_compiled, compiled_src = make_compiled_forward(len(Ws_py))
    if name == "small":
        print("compiled source (small 예시):")
        for line in compiled_src.strip().splitlines():
            print(f"    {line}")

    # 정합성
    ref_a = forward_listlist(xs_py[0], Ws_py, bs_py)
    ref_a2 = forward_array(xs_arr[0], Ws_arr, bs_arr)
    ref_b = forward_c_scalar(xs_t[0], Ws_t, bs_t).to_list()
    ref_c = forward_c_vec(xs_t[0], Ws_t, bs_t).to_list()
    ref_d = call_compiled(forward_compiled, xs_t[0], Ws_t, bs_t).to_list()
    diff = {
        "A→A'": max_abs_diff(ref_a, ref_a2),
        "A→B": max_abs_diff(ref_a, ref_b),
        "A→C": max_abs_diff(ref_a, ref_c),
        "A→D": max_abs_diff(ref_a, ref_d),
    }

    # ─── (1) 단계별 forward 시간 분포 ─────────────────────────────────
    print(f"\n── 단계별 forward 시간 분포 ({n:,} samples) ──────────────────")
    print(f"  {'단계':<32}{'min':>9}{'p50':>9}{'mean':>9}{'p99':>9}{'std':>9}{'CoV':>7}")
    print("  " + "─" * 84)

    results: Dict[str, Tuple[Stats, float]] = {}

    if not skip_pure_python:
        times = time_one_pass(forward_listlist, xs_py, Ws_py, bs_py)
        s = Stats(times)
        results["A  Pure Python (list)"] = (s, 0.0)
        print_stage(f"A  Pure Python (list)", s)

        times = time_one_pass(forward_array, xs_arr, Ws_arr, bs_arr)
        s = Stats(times)
        results["A' Pure Python (array.array)"] = (s, diff["A→A'"])
        print_stage(f"A' Pure Python (array.array)", s)

    times = time_one_pass(forward_c_scalar, xs_t, Ws_t, bs_t)
    s = Stats(times)
    results["B  C scalar (no-SIMD)"] = (s, diff["A→B"])
    print_stage(f"B  C scalar (no-SIMD)", s)

    times = time_one_pass(forward_c_vec, xs_t, Ws_t, bs_t)
    s = Stats(times)
    results["C  C vectorized"] = (s, diff["A→C"])
    print_stage(f"C  C vectorized", s)

    times = time_one_pass_compiled(forward_compiled, xs_t, Ws_t, bs_t)
    s = Stats(times)
    results["D  C vec + fused"] = (s, diff["A→D"])
    print_stage(f"D  C vec + fused", s)

    # 단계별 gap 정리
    print(f"\n── 단계 gap 별 기여 (median 기준) ──────────────────")
    labels = list(results.keys())
    for i in range(1, len(labels)):
        prev_s, _ = results[labels[i - 1]]
        cur_s, _ = results[labels[i]]
        reduction = (1 - cur_s.p50 / prev_s.p50) * 100
        gap_name = _gap_name(labels[i - 1], labels[i])
        print(
            f"    {labels[i - 1][:3]} → {labels[i][:3]:<4}  "
            f"{gap_name:<24}  "
            f"{fmt_time(prev_s.p50)} → {fmt_time(cur_s.p50)}   "
            f"{reduction:>6.2f}% 단축"
        )

    # ─── (2) 연산별 호출 통계 (C vec 경로) ────────────────────────────
    print(f"\n── 연산별 호출 통계 (C vec, {n:,} forward 안의 각 op) ──────────────────")
    print(f"  {'연산':<24}{'호출수':>8}{'min':>9}{'p50':>9}{'mean':>9}{'p99':>9}{'std':>9}")
    print("  " + "─" * 80)
    op_times = profile_per_op_c_vec(xs_t, Ws_t, bs_t)
    op_total = 0.0
    for op, times in op_times.items():
        s = Stats(times)
        op_total += s.mean * len(times)
        print(f"  {op:<24}{len(times):>8}"
              f"{fmt_time(s.min):>9}{fmt_time(s.p50):>9}{fmt_time(s.mean):>9}"
              f"{fmt_time(s.p99):>9}{fmt_time(s.std):>9}")
    print(f"  {'-- 합계 (mean × 호출수)':<32}                            {fmt_time(op_total):>9}")

    print(f"\n── 연산별 호출 통계 (Fused, {n:,} forward 안) ──────────────────")
    print(f"  {'연산':<24}{'호출수':>8}{'min':>9}{'p50':>9}{'mean':>9}{'p99':>9}{'std':>9}")
    print("  " + "─" * 80)
    op_times = profile_per_op_fused(xs_t, Ws_t, bs_t)
    op_total = 0.0
    for op, times in op_times.items():
        s = Stats(times)
        op_total += s.mean * len(times)
        print(f"  {op:<24}{len(times):>8}"
              f"{fmt_time(s.min):>9}{fmt_time(s.p50):>9}{fmt_time(s.mean):>9}"
              f"{fmt_time(s.p99):>9}{fmt_time(s.std):>9}")
    print(f"  {'-- 합계 (mean × 호출수)':<32}                            {fmt_time(op_total):>9}")


def _gap_name(prev_label, cur_label):
    prev = prev_label[:2].strip()
    cur = cur_label[:2].strip()
    return {
        ("A", "A'"): "데이터 레이아웃",
        ("A'", "B"): "네이티브 루프 (C)",
        ("B", "C"): "Vector 최적화 (SIMD)",
        ("C", "D"): "표현식 융합",
        ("A", "B"): "C-Level (포괄)",  # pure_python skip 시
    }.get((prev, cur), "?")


def print_stage(label, s: Stats):
    print(
        f"  {label:<32}"
        f"{fmt_time(s.min):>9}{fmt_time(s.p50):>9}{fmt_time(s.mean):>9}"
        f"{fmt_time(s.p99):>9}{fmt_time(s.std):>9}"
        f"{s.cov * 100:>6.2f}%"
    )


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("=" * 92)
    print(" Pydya 종합 추론 KPI 벤치마크 (단계 + 연산별 호출 통계)")
    print("=" * 92)
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
        spec = MODELS[size]
        run_model(size, spec)

    print(f"\n{'=' * 92}")
    print(" 해석 가이드:")
    print("   - A → A' (데이터 레이아웃): list-of-list 의 nested subscript 와 평탄")
    print("     array.array 의 차이. Python loop 가 dominate 라 차이 작음(보통 0~20%).")
    print("   - A' → B (네이티브 루프): 같은 알고리즘을 Python 인터프리터 → C 핫")
    print("     루프로. **여기서 압도적 가속 (수십x)** — 우리 컴파일러의 본진.")
    print("   - B → C (SIMD): scalar baseline 도 -O3 + unroll 살아있어 큰 차이 X.")
    print("     큰 행렬에서 memory-bound 라 SIMD 효과 더 줄어듦.")
    print("   - C → D (융합): matmul 비중에 반비례. 작은 모델 더 의미, 큰 모델")
    print("     은 matmul 이 시간을 다 차지해 fusion 효과 0 수렴.")
    print("   - 연산별 표: forward 안에서 시간이 *어디로* 가는지 정량.")
    print("     fusion 의 효과가 작은 모델에선 add+relu 비중이 크고, 큰 모델은")
    print("     matmul 압도라는 사실이 표 안 호출수 × 시간 분포에서 직접 보임.")
    print("=" * 92)


if __name__ == "__main__":
    main()
