# Pydya 성능 — 무엇을 측정했고, 왜 이런 숫자가 나오는가

이 문서는 `benchmarks/` 의 각 벤치마크 결과를 *왜 그 수치인지* 함께 정리한다.
숫자만 보면 "fusion 이 1.18x 인데 의미 있나?" 같은 질문에 흔들리지만, *왜*
까지 알면 그 수치가 우리 컴파일러의 어떤 한계/특성을 정확히 가리키는지
이해할 수 있다.

모든 벤치는 4코어 i5, 컨테이너 환경 (Python 3.11.15) 기준. 본문의 수치는
median(p50). 분포는 각 벤치마크 출력에 포함된다.

---

## 1. C Tensor 단일 연산 (`bench_tensor_ops.py`)

**측정한 것**: pydya.Tensor 의 element-wise 연산과 matmul 을 Python list 기반
순수 Python 구현과 비교. N 을 sweep.

**실측**:

| 연산 | N=1K | N=10K | N=100K | N=1M |
|---|---|---|---|---|
| element-wise add | 116x | 266x | 135x | 90x |
| element-wise sub | 115x | 264x | 128x | 89x |
| element-wise mul | 112x | 261x | 152x | 88x |
| matmul (2D × 1D) | (32×64) 51x | (64×256) 40x | (128×512) 37x | (1024×1024) 37x |

**이유**:
- *왜 이렇게 큰가* — Python 의 `[x+y for x,y in zip(...)]` 은 매 원소마다
  PyObject 박싱/언박싱, dict-lookup, ref count 조정, dispatch 다 일어남.
  C 안은 raw float arithmetic 한 줄. *인터프리터 vs 컴파일된 핫루프* 의
  근본 격차 (대략 50-200배).
- *왜 N=10K 에서 피크* — L1/L2 cache 에 다 들어가 memory-bound 영향 최소.
- *왜 N=1M 에서 떨어짐* — 4MB > L2 보통 크기. DRAM 대역폭이 천장. C 쪽은
  여전히 빠르지만 Python 도 *데이터 cache miss 가 같은 비율로* 느려지지
  않아서 (이미 PyObject 비용이 dominate) 상대 격차가 줄어든다.
- *matmul 의 일정한 ~37-50x* — element-wise 와 달리 inner loop 가 매우
  tight (`acc += W[i][j]*x[j]`). Python list-of-list 의 nested subscript +
  ref count 가 매 inner iter 마다 발생. 상수 가까운 비율로 격차 유지.

**한계 / Trade-off**:
- 우리 matmul 은 **naive** — tiling/blocking 없음. 큰 행렬에서 memory-bound
  됨. BLAS 급 구현은 tile/register block 으로 cache 를 채워가며 돈다.
- 우리는 *컴파일러 프로젝트* 라 BLAS 구현은 의도적으로 스코프 외. 다만
  이 한계가 다음 두 가지 벤치 (SIMD 효과 / fusion 효과) 의 천장도 결정함.

---

## 2. 표현식 융합 `madd` — element-wise (`bench_madd.py`)

**측정한 것**: `a*b + c` 미융합 (Tensor 연산자 체인) vs 융합 `madd(a, b, c)`.
N 을 sweep, p99/std 까지.

**실측**:

| N | UNFUSED | FUSED | speedup |
|---|---|---|---|
| 1,000 | 602ns | 423ns | 1.43x |
| 10,000 | 3.11us | 2.01us | 1.55x |
| 100,000 | 82.4us | 59.5us | 1.38x |
| 1,000,000 | 1.075ms | 686us | 1.57x |
| 4,000,000 | 22.6ms | 4.38ms | **5.17x** |

p99/std 도 fused 가 일관되게 작음 (alloc 변동성 제거).

**이유**:
- *왜 ~1.5x 가 기본* — 미융합은 ① `a*b` → 임시 텐서 1개 alloc + write,
  ② `+ c` → 임시 텐서 2개 alloc + write. 두 패스 + 두 alloc. 융합은 한
  패스 + 한 alloc. 메모리 traffic 이론적으로 24N → 16N (~33% 감소). 실측
  1.4~1.6x 는 이 33% 와 alloc 비용 절감의 합.
- *왜 N=4M 에서 5x* — 텐서 16MB. 임시 텐서 두 개를 만들면 작업 데이터가
  ~48MB 로 L3 (보통 8-16MB) 초과 → DRAM 까지 오감. 융합은 하나만
  유지하니 L3 안에 거의 머묾. **memory bandwidth wall 을 넘느냐 안 넘느냐의
  차이**가 5x 로 폭증.
- *왜 p99/std 가 작음* — 임시 alloc 은 OS page fault, malloc heap 상태,
  GC 마다 변동성. 한 번 alloc 이 그 변동성을 1/3 로 줄임.

**한계**: 우리 fusion 은 *단일 expression* (a*b+c 한 줄) 만. 그래프-level
fusion (여러 layer 묶음) 은 미구현. 그게 Optimium 의 30% 가속 영역.

---

## 3. 표현식 융합 `linear_relu` — Dense+ReLU (`bench_linear_relu.py`)

**측정한 것**: `relu(W @ x + b)` 미융합 vs 융합 `linear_relu(W, x, b)`.
hidden size 를 8 → 2048 sweep.

**실측 패턴**:
- h ≤ 64: 1.0-1.2x (절대 시간 짧아 측정 노이즈 영역)
- h ≥ 128: 안정적으로 **1.15-1.20x**

**이유**:
- *이득의 출처는 출력 버퍼 메모리 traffic* — 미융합은 (matmul 결과 8KB) →
  (+b 임시 8KB) → (relu 결과 8KB) 세 번 출력 크기 메모리 패스. 융합은
  한 번. matmul 자체 (W 읽기) 는 양쪽 동일.
- *왜 hidden 커져도 1.18x 유지* — matmul 시간이 같이 커지지만 출력 traffic
  도 같이 커져 *비율* 이 거의 일정. 절대 절감은 hidden 클수록 큼.

**한계 / 천장**: 단일 expression fusion 의 천장이 ~1.2x. 이걸 넘으려면
matmul 자체를 빠르게 (우리 스코프 밖) 또는 graph-level fusion (여러
레이어 한 커널로, 코드 생성 필요).

---

## 4. `attr[unroll]` — Python-level (`bench_unroll.py`)

**측정한 것**: 같은 워크로드를 `attr[{'unroll': True}]` 유무로 컴파일해
실행 시간 비교.

**실측**:
- 워크로드 A — i-의존 piecewise activation: **1.54x**
- 워크로드 B — i-modular 가중합 (`i % 3` 분기): **2.51x**

**이유**:
- *왜 1.5-2.5x 까지* — unroll 시 매 반복마다 일어나는 `FOR_ITER`, `i` 의
  STORE/LOAD, 그리고 무엇보다 `if i < K` 같은 *i-의존 분기* 가 컴파일
  타임에 상수로 접혀 사라짐. 워크로드 B 의 `i % 3 == 0/1/2` 는 매 반복마다
  modulo + compare + jump 3 연쇄였는데 unroll 후 그냥 직선 코드.

**중요한 한계**: 이 가속은 *Python 인터프리터의 분기/iter 옵코드 비용을
없애는 데서만* 나옴. 핫코드가 C 안에 있는 워크로드 (텐서 추론 등) 에서는
runtime 기여 **0**. attr[unroll] 은 *일반 가속기가 아니라 부분평가가 의미
있는 한정 케이스 도구*.

**우리 컴파일러에서의 의미**: substrate. 펼친 코드가 다른 fold/dce/branch
패스가 더 단순화할 수 있는 기회를 제공.

---

## 5. `attr[parallel]` — 자동 병렬화 (`bench_parallel.py`)

**측정한 것**: 다양한 워크로드 (per-item 비용) × 워커수 별 가속, 3.11 (스레드풀)
과 3.14 (서브인터프리터) 양쪽.

**실측** (4코어):

| 워크로드 | 3.11 threadpool w=4 | 3.14 subinterp w=4 |
|---|---|---|
| A) 가벼움 (`i*i`) | 0.02x (오히려 손해) | 0.00x (cold start 압도) |
| B) 중간 (sum 10K) | 0.93x | 0.22x |
| C) 무거움 (sum 2M) | 1.05x (GIL) | **2.76x** |

**이유**:
- 3.11 의 ThreadPoolExecutor 는 GIL 공유 — 순수 파이썬 본문은 직렬화돼
  본문이 무거워도 코어수 대비 가속 안 나옴 (1.0-1.06x). GIL 풀어주는 작업
  (numpy/Tensor C 호출) 만 진짜 가속.
- 3.14 의 `concurrent.interpreters` 는 **인터프리터별 GIL** (PEP 684) — 진짜
  멀티코어 가능. 무거운 워크로드 C 에서 4코어 ~2.76x (이상적 4x 의 69%).
- cold start 비용: 서브인터프리터 생성 + 데이터 직렬화 ~수십 ms. 가벼운
  워크로드는 이 비용에 묻혀 오히려 손해. 본문이 충분히 무거워야 이득.

**한계**: 우리 백엔드는 본문을 *expr 소스 + 캡처 dict* 로 직렬화해 서브
인터프리터에 보냄. 캡처가 picklable 이어야 하고, 자유변수 무거우면 IPC
비용 증가. *언제 attr[parallel] 을 붙일 가치가 있는가* = per-item 본문이
수십 ms 이상.

---

## 6. 종합 추론 break-down (`feature_breakdown_benchmark.py`)

**측정한 것**: MLP forward 를 네 단계로 분해:
- A) Pure Python
- B) C scalar (auto-vectorize 끈 변종)
- C) C vectorized
- D) C vec + fused

**실측** (huge: 3-layer 784→2048→1024→10, N=3000):

| 단계 | 시간 | vs A |
|---|---|---|
| A) Pure Python | 420.8s | 1.0x |
| B) C scalar | 7.76s | **54x** |
| C) C vectorized | 7.54s | 55.8x |
| D) Fused | 7.53s | 55.9x |

기여도 (gap 분석): C-Level Tensor **98%**, SIMD 2.8%, Fusion 0.1%.

**이유**:
- *왜 C-Level Tensor 가 압도적* — 1번 결과의 연장. Python 인터프리터 우회
  + raw C 핫루프 + contiguous 메모리 + alloc 1회 다 묶음.
- *왜 큰 모델일수록 SIMD 효과 작음* — matmul 이 memory-bound 됨. SIMD 의
  연산량 가속이 memory bandwidth 천장에 막힘. + scalar baseline 도 `-O3
  -funroll-loops` 살아있어 강한 baseline.
- *왜 fusion 0.1% 인지* — matmul 이 시간의 99%. 1% 의 add/relu 부분에서
  fusion 효과 ~10% 절감 ≈ 전체 0.1%. **이건 우리 단일 expression fusion
  의 구조적 천장.**

**전체 narrative**: 우리 컴파일러의 진짜 큰 기여는 **C-level 핫루프로의
이동**. SIMD/fusion 은 위에 얹는 가산점인데, 작은 행렬에서 더 보임. 큰
모델에서 더 큰 가속을 원하면 *(a) matmul 자체를 BLAS 급으로* 또는 *(b)
그래프-level fusion* — 둘 다 컴파일러 영역 밖 (a) 또는 한참 큰 작업 (b).

---

## attr[unroll] 의 정직한 위치

attr[unroll] 은 추론 hot path 에 *runtime 기여 0*. 그러나 *부분평가
substrate* 로서 의미:
- 펼친 본문이 dce/branch/inline 패스에 추가 단순화 기회 제공
- Python 인터프리터 안에 분기 많은 작은 루프가 있는 워크로드 (bench_unroll
  의 워크로드 A/B) 에서는 1.5-2.5x 의 실효 가속

**이게 일반 가속기가 아님을 사용자가 알게 하는 것이 중요.** 추론 워크로드
에 attr[unroll] 을 붙여도 효과 없음.

---

## 우리 숫자가 Nadya 의 1/4 에 못 미치는 부분 — 구조적 이유

Nadya 의 fusion 사례 (탐구3 의 conv+ReLU): 그래프 단 ~1.3x 가속. 우리
linear_relu fusion: 1.0-1.2x. 차이의 이유:

1. **Nadya 는 graph-level fusion** — 여러 layer 묶음. 우리는 단일 expression.
2. **Nadya 의 baseline (그래프 인터프리터) vs 우리 baseline (-O3 C)** —
   비교 baseline 의 강도 차이.
3. **Nadya 의 워크로드는 conv 중심** — 다양 op 비중. 우리는 matmul 이 99%.
4. **Nadya 는 MLIR + 코드 생성** — 우리는 Python-to-Python 부분평가.

이건 *Nadya 의 기법이 더 강한 게 아니라, 우리가 그 기법이 작동할 substrate
(graph-level IR, 다양한 op 비중, 잘 짜인 baseline kernel) 자체를 안 갖췄다*
는 의미. 우리 SIMD/Fusion 자체는 정상 작동, **베이스가 약하면 위에 얹는
최적화 효과도 같이 작아진다**.

---

## 정리

- C-Level Tensor 는 *항상 큰 기여* (~50x). 우리 컴파일러의 본진.
- SIMD 는 작은 모델에서 잘 보임 (12-14%), 큰 모델에서 작아짐 (3-5%).
- 단일 expression Fusion 은 작은 모델/체인에서 잘 보임 (madd 5x, linear_relu
  1.18x), matmul 지배적 모델에선 작음 (0.1%).
- attr[unroll] 은 *부분평가 substrate*, 분기 많은 작은 Python 루프에서만
  1.5-2.5x 의 직접 가속.
- attr[parallel] 은 무거운 본문 + 3.14 에서 코어수에 가까운 가속.

각 숫자가 어떤 한계를 가리키는지 명확히 알면, 다음 한 발이 무엇인지도
명확해진다.
