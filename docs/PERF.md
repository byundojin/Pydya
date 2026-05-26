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

## 6. 종합 추론 KPI (`feature_breakdown_benchmark.py`)

프로젝트의 KPI. **MLP forward 의 기능별 분해 + 연산별 호출 통계 + 이론합 검증**.

### 측정 방식

모든 측정 — `_stats.time_stable` (adaptive inner repeat 으로 표본당 ≥ 100us/500us 보장) 으로 CoV < 15% 안정 측정. *표본 수 × inner × 총 호출수* 가 출력에 명시.

4단계 분해 (`A → B → C → D`):
- A — Pure Python (list-of-list)
- B — C Tensor scalar (auto-vec 끈 변종)
- C — C Tensor vectorized
- D — Compiled (fused linear_relu)

각 gap 의 의미:
- `A → B` = C 레벨 핫루프 (Python 인터프리터 → C native loop)
- `B → C` = Vector 최적화 (auto-vectorize / SIMD)
- `C → D` = 표현식 융합 (relu(W@x+b) → linear_relu)

추가: **연산별 단독 측정** — 사전 계산된 입력으로 matmul/add/relu/linear_relu 를 격리해 분포 측정. 합산이 stage 측정과 일치하는지 sanity check.

### 실측 (4코어 i5, mean 기준)

| 모델 | A→B (C 핫루프) | B→C (SIMD) | C→D (Fusion) | 이론합↔실측 |
|---|---|---|---|---|
| small (64→32→10) | **37.6x** (97.3%) | 1.10x (9.5%) | **1.41x** (28.9%) | diff 27.7% |
| medium (256→128→10) | 31.6x (96.8%) | **2.20x** (54.6%) | 0.99x (-1.2%) | diff 4.7% |
| large (784→1024→10) | 25.8x (96.1%) | **1.99x** (49.7%) | 1.01x (0.7%) | **diff 0.7%** |
| huge (3-L 784→2048→1024→10) | 25.4x (96.1%) | **1.95x** (48.7%) | 1.02x (2.4%) | diff 6.2% |

### 측정 근거 (왜 이 수치인가)

**A (Pure Python)** — 모든 모델에서 **32-37 ns/op** (matmul mul-add 당). CPython 산술 op 평균 ~30-50ns/op (BINARY_OP + BINARY_SUBSCR + STORE) 와 정확히 일치. *측정이 인터프리터 비용으로 완전히 설명됨.*

**C (C vec)** — 모든 모델에서 **0.47-0.85 ns/op** = ~1.6-2.8 cycle/op @ 3GHz. 1-3 cycle/op 영역은 *SIMD 활성 + memory-bound 중간* — 즉 AVX2 가 일부 동작하지만 천장은 메모리 대역폭.

**SIMD 가 size 별로 다른 이유**:
- **small** 1.10x: 매트멀 32×64 = 2K ops. C 핫루프 자체가 1us 미만. 호출 오버헤드/측정 노이즈 비중이 SIMD 이득보다 큼.
- **medium 이상** 1.95-2.20x: 매트멀이 cache 친화 크기 (L1-L2 영역). SIMD 진짜로 동작.
- 이론 AVX2 8x 대비 ~1/4 인 이유: (1) scalar baseline 도 `-O3 -funroll-loops` 라 강함, (2) memory-bound 비중이 큼.

**Fusion 이 size 별로 다른 이유**:
- **small** 1.41x: forward 시간이 ~2us 인데 add+relu 호출/alloc 오버헤드가 그 중 ~600ns. 절감 비중 큼.
- **medium 이상** 1.0x: matmul 시간이 forward 의 95%+. add/relu 오버헤드는 절대값으론 작지 않지만 *상대 비중* 1% 미만. fusion 절감의 천장.

**이론합↔실측 diff 가 size 별로 다른 이유**:
- **large 0.7%** 가 가장 정확 — 매트멀이 compute-bound 라 op 단독 측정과 forward 안 측정이 거의 같음.
- **small 27.7%** 큰 갭 — forward 가 너무 짧아 (us 단위) 측정 노이즈가 큼. CoV 와 일관.

### 연산별 호출 통계 — *시간이 어디로 가는가*

large 모델 (784→1024→10) 예시, 표본 200× adaptive inner:

| 연산 | 표본×inner = 총호출 | min | p50 | mean | p99 | std | CoV |
|---|---|---|---|---|---|---|---|
| matmul L1 (1024×784) | 200×1 = 200 | 426us | 501us | **504us** | 573us | 29us | 5.7% |
| add L1 (1024,) | 200×289 = 57,800 | 142ns | 160ns | 167ns | 281ns | 23ns | 13.5% |
| relu L1 (1024,) | 200×324 = 64,800 | 136ns | 139ns | 165ns | 390ns | 58ns | 35.1% |
| linear_relu L1 | 200×1 = 200 | 426us | 485us | **487us** | 549us | 27us | 5.6% |
| matmul L2 (10×1024) | 200×16 = 3,200 | 5.7us | 6.3us | 6.6us | 10us | 822ns | 12.4% |
| add L2 (10,) | 200×309 = 61,800 | 98ns | 109ns | 111ns | 198ns | 19ns | 16.9% |

→ matmul L1 가 forward 의 **99%** (504us / 508us). 나머지 모든 op 합쳐 4us. *fusion 이 큰 모델에서 1.01x 인 이유가 표에 그대로* — linear_relu 가 matmul L1 자체 시간을 504us → 487us 로 ~3.4% 줄여 전체 0.7% 단축. 더 줄일 여지가 *구조적으로 없음.*

### 한계 / 다음

- **graph-level fusion 없음** — 단일 expression 한정. Optimium 의 conv+ReLU 같은 그래프 묶음 융합으로 가야 더 큰 효과.
- **matmul 이 naive** — tiling/blocking 없음. 큰 모델에서 memory-bound. SIMD 천장도 여기.
- 위 한계 그대로 두는 게 Python-to-Python 컴파일러 정체성 유지하는 trade-off.

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
