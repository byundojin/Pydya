# Pydya

Pydya 는 Python 소스를 **부분 평가(partial evaluation)** 하는 소스-투-소스
컴파일러다. 일부 값이 컴파일 시점에 고정되어 있다고 알려 주면, 그 값에
의존하는 계산을 미리 수행하고 결정된 분기·함수 호출을 펼쳐, 더 단순하고
빠른 잔여(residual) 프로그램을 만들어 낸다.

설계는 ENERZAi 의 [Nadya](https://medium.com/@enerzai) 에서 영감을 받았다.
정적으로 알려진 값은 컴파일 타임에 처리하고, 안전한 한도 내에서 병렬화·
펼침·표현식 융합 같은 메타프로그래밍 변환까지 수행한다. C 확장을 통해
일부 핫루프는 GIL 을 풀고 컴파일러 자동 벡터화의 이득까지 가져온다.

## 한눈에 보는 기능

| 기능 | 진입점 | 한 줄 설명 |
|---|---|---|
| 컴파일 타임 변수 | `CompileVar[T]()` / `CompileVar('name')` | Nadya `template<>` 대응. 환경 값으로 정적 단형화 |
| 함수 단위 특수화 | `@specialize(env)` | 데코레이터 한 줄로 함수 부분 평가 |
| 모듈 import 특수화 | `from pydya.importer import specialize_here` | `from __future__` 식 모듈 자기 opt-in |
| 자동 병렬화 | `attr[{'parallel': True}]` 마커 | 독립 map 을 서브인터프리터/스레드풀로 lowering |
| 루프 펼침 | `attr[{'unroll': True}]` 마커 | 정적 `range` 의 본문을 펼침(부분평가 substrate) |
| C 텐서 primitive | `pydya.Tensor` | **N-D** float32, GIL 해제, 자동 벡터화, `@` 연산자 |
| 표현식 융합 | `: Tensor` 어노테이션 신뢰 | `a*b+c` → `madd`, `relu(W@x+b)` → `linear_relu` |
| 신경망 추론 | `pydya._tensor.{matmul, relu, linear_relu}` | 사전학습 MLP 의 forward pass — XOR/손글씨 숫자 분류 |

## 설치

```bash
pip install -e .
```

Python **3.11 이상** 필요. C 확장(`pydya._tensor`)을 같이 빌드하려면 시스템에
C 컴파일러와 Python 개발 헤더가 있어야 한다. 별도로 빌드하려면:

```bash
python setup.py build_ext --inplace
```

3.14 에서 빌드하면 진짜 멀티코어 병렬(서브인터프리터, 인터프리터별 GIL)
백엔드가 자동 활성화된다.

## 빠른 시작 — 부분 평가

컴파일 시점에 고정하고 싶은 값을 `CompileVar` 로 선언한다.

```python
from pydya import CompileVar, compile_source

source = '''
V = CompileVar[int]()

def f(a):
    return a + V

if V < 5:
    a = f(5)
else:
    a = 5

b = a + V
print(b)
'''

print(compile_source(source, env={"V": 3}))
```

출력:

```python
def f(a):
    return a + 3
a = 8
b = a + 3
print(b)
```

`V` 가 모두 `3` 으로 접히고, `f` 는 `return a + 3` 으로 특수화되며,
`if V < 5` 는 `True` 로 접혀 `else` 가지가 사라진다.

## 기능별 사용 예시

### `@specialize` — 함수 단위 부분 평가

```python
from pydya import CompileVar, specialize

@specialize({'V': 3})
def f(a):
    V = CompileVar[int]()
    return a + V

print(f(10))                 # 13
print(f.__pydya_source__)    # "def f(a):\n    return a + 3"
```

### import 시점 특수화 (Nadya `__future__` 식 디렉티브)

```python
# kernel.py 상단에 마커 한 줄로 opt-in
from pydya import CompileVar
from pydya.importer import specialize_here

SCALE = CompileVar[int]()

def scaled(x):
    return x * SCALE
```

```python
# main.py — 대상 import 이전에 환경 등록
import pydya.importer as importer
importer.install({'SCALE': 4})
import kernel
print(kernel.scaled(10))   # 40
```

### `attr[parallel]` — 자동 병렬 map

```python
from pydya import attr

out = [0] * 8

attr[{'parallel': True, 'workers': 4}]
for i in range(8):
    out[i] = i * i + 10
```

`compile_source` 가 위 마커를 인식해 `pydya.runtime.parallel_map_into` 호출로
lowering 한다. 본문이 *반복 간 파괴적 갱신 없는 독립 map* 일 때만 허용
(Nadya 의 destructive update 회피 규칙과 동일). 백엔드는 다음 순서로
자동 선택된다.

1. `concurrent.interpreters` (Python 3.14+) — **인터프리터별 GIL, 순수 파이썬 본문도 멀티코어**
2. `ThreadPoolExecutor` — 모든 버전에서 동작, GIL 푸는 작업(C/numpy/Tensor) 만 가속
3. 직렬 폴백

### `attr[unroll]` — 컴파일 타임 루프 펼침

```python
from pydya import CompileVar, attr, compile_source

source = '''
from pydya import attr
W = CompileVar[int]()

def dot_product(a, b):
    result = 0
    attr[{'unroll': True}]
    for i in range(W):
        result += a[i] * b[i]
    return result
'''

print(compile_source(source, env={'W': 4}))
```

출력:

```python
def dot_product(a, b):
    result = 0
    result += a[0] * b[0]
    result += a[1] * b[1]
    result += a[2] * b[2]
    result += a[3] * b[3]
    return result
```

> **정직한 한계**: 이 unroll 은 *부분평가 substrate* 일 뿐, CPython 바이트코드
> VM 위에선 런타임 가속이 거의 없다. Optimium 이 unroll 로 얻는 이득(메모리
> 접근 감소, SIMD, ILP, register reuse)은 모두 네이티브 바이너리 컴파일을
> 전제로 한다. 우리에게서 펼침은 다음 단계인 C 텐서 + 표현식 융합의
> 패턴 매칭을 가능하게 하는 발판이다.

### C 텐서 primitive (N-D)

```python
from pydya import Tensor

# 1D
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([10.0, 20.0, 30.0, 40.0])
print((a * b).to_list())                 # [10.0, 40.0, 90.0, 160.0]

# 2D (행렬)
W = Tensor([[1.0, 2.0], [3.0, 4.0]])
x = Tensor([5.0, 6.0])
print(W.shape)                            # (2, 2)
print((W @ x).to_list())                  # [17.0, 39.0]  — matmul (2D × 1D)
```

float32 contiguous row-major, 64바이트 정렬. 산술 핫루프는 `Py_BEGIN_ALLOW_THREADS`
로 GIL 을 풀고 `restrict` 포인터 + `-O3 -march=native` 로 컴파일러가 SIMD
명령어를 깐다. 모듈 함수: `matmul(W, x)`, `relu(t)`, `linear_relu(W, x, b)`,
`madd(a, b, c)`.

### 표현식 융합 — `: Tensor` 어노테이션 신뢰

```python
from pydya import compile_source

source = '''
def fma(a: Tensor, b: Tensor, c: Tensor):
    return a * b + c
'''

print(compile_source(source))
```

출력:

```python
import pydya._tensor as __pydya_t

def fma(a, b, c):
    return __pydya_t.madd(a, b, c)
```

미융합 `a*b+c` 는 임시 텐서 2개를 만들고 메모리를 3회 추가 순회한다(~24N
bytes traffic). 융합 호출은 단일 할당 + 단일 메모리 순회(~16N bytes). 추가로
컴파일러가 FMA 명령어(`vfmadd`)를 emit 할 수 있어 정밀도(단일 라운딩)까지
향상된다.

`Linear + ReLU` 패턴도 같은 결로 `linear_relu` 융합으로 lowering 된다
(Optimium 탐구3 의 Conv+ReLU 융합 대응):

```python
def step(W: Tensor, x: Tensor, b: Tensor):
    return relu(W @ x + b)
```
↓
```python
import pydya._tensor as __pydya_t
def step(W, x, b):
    return __pydya_t.linear_relu(W, x, b)
```

### 신경망 추론 — XOR 부터 손글씨 숫자까지

위 융합 위에 그대로 작은 MLP 추론을 얹는다. 학습은 외부에서 끝낸 가중치를
JSON 으로 받아 pydya Tensor 로 올린다.

```python
def forward(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
    h = relu(W1 @ x + b1)        # → linear_relu(W1, x, b1) 로 융합
    return W2 @ h + b2           # raw logits (argmax 가 softmax 를 보존)
```

* **XOR 진리표** — `examples/xor_inference.py` 4/4 통과 (하드코딩 가중치)
* **손글씨 숫자 (8×8)** — `examples/digit_inference.py`, sklearn 의 `load_digits`
  로 외부 학습한 `64 → 32 → 10` MLP. held-out 50 샘플 기준 **98% 정확도**

## 측정 결과

벤치마크는 `benchmarks/` 에서 실행할 수 있다. 분포(min/p50/p95/p99/std) 까지
같이 본다. 4코어 i5 기준.

| 측정 | 결과 |
|---|---|
| C Tensor element-wise (`bench_tensor_ops.py`, N=1K~1M vs Python list) | **88~266x** |
| C Tensor matmul (2D × 1D, 다양 shape) | **37~51x** |
| 표현식 융합 `a*b+c` (`bench_madd.py`, N=1K~4M) | **1.4~5.2x** (N 클수록 큼) |
| `linear_relu` 융합 (`bench_linear_relu.py`, hidden=128~2048) | **1.15~1.20x** |
| `attr[unroll]` (`bench_unroll.py`, 분기 많은 작은 루프) | **1.5~2.5x** |
| `attr[parallel]` (3.14 서브인터프리터, 무거운 본문, 4코어) | **2.76x** (이상적 4x 의 69%) |
| `attr[parallel]` (3.11 스레드풀, 동일 본문) | 1.05x (GIL) |
| 손글씨 숫자 추론 (`examples/digit_inference.py`) | **98% 정확도** (50샘플) |
| 종합 추론 4단계 분해 (`feature_breakdown_benchmark.py`, huge 3-layer) | C-Level 98%, SIMD 2.8%, Fusion 0.1% |

**각 숫자가 왜 그렇게 나오는지** 는 [`docs/PERF.md`](docs/PERF.md) 에 정리.

## 패스 파이프라인

`compile_source` 는 소스를 AST 로 파싱한 뒤 다음을 순서대로 적용한다.

| 순서 | 패스 | 모듈 | 역할 |
|------|------|------|------|
| 1 | collect | `passes/collect.py` | `CompileVar` 선언과 compile-only `pydya` import 를 제거하고 정적 환경 수집 |
| 2 | fold | `passes/fold.py` | 정적 이름 치환 + 상수 산술/비교/불리언/단항 폴딩 |
| 3 | parallelize | `passes/parallelize.py` | `attr[{...}]` 마커 일괄 소비. `parallel` 키는 호출로 lowering, `unroll` 키는 다음 for 에 플래그 부착 |
| 4 | unroll | `passes/unroll.py` | 플래그 붙은 for 의 본문을 `range` 인자별로 펼침 (안전 검사 포함) |
| 5 | branch | `passes/branch.py` | 상수로 접힌 `if`/`while`/조건식의 선택된 가지만 유지 |
| 6 | inline | `passes/inline.py` | 상수 인자로 호출되는 단순 함수를 인라인·특수화 |
| 7 | dce | `passes/dce.py` | 함수 본문 내 부작용 없는 죽은 대입 고정점 제거 |
| 8 | fuse_tensors | `passes/fuse_tensors.py` | `: Tensor` 어노테이션 신뢰. `a*b+c` → `madd`, `relu(W@x+b)` → `linear_relu` 융합. 함수 본문 내 지역변수 Tensor 전파, bare `matmul`/`relu`/`madd`/`linear_relu` 이름 자동 qualify |

## 설계상 결정과 한계

부분평가는 *관찰 가능한 동작을 바꾸지 않을 때* 만 변환한다. Pydya 는 안전을
위해 보수적으로 동작한다.

- **정적 이름만 치환한다.** 일반 변수의 우변이 상수로 계산되더라도 사용처에
  전파하지 않는다.
- **상수 인자 호출만 인라인한다.** 런타임 식 인라인은 부작용 중복·순서
  변경 위험이 있어 하지 않는다.
- **`attr` 마커는 trust-based.** `: Tensor` 어노테이션이나 `attr[parallel]`
  의 안전성 약속을 컴파일러가 검증하지 않고 사용자 의지로 받아들인다
  (Pydya 가 `attr[parallel]` 의 destructive update 검사는 수행).
- **`attr[unroll]` 단독 런타임 가속은 없다.** 부분평가의 substrate
  역할이며, 진짜 가속은 C 텐서 + 융합과 결합할 때 따라온다.
- **Tensor 는 N-D float32 까지** 지원 (`shape`/`ndim`/`@` 연산자 + matmul 2D×1D).
  다른 dtype/일반 broadcast/2D×2D 배치 matmul/conv 는 미구현.
- **융합 패턴 제한** — `a*b+c` (madd), `relu(W@x+b)` 의 세 가지 모양 (`W@x` /
  `matmul(W,x)` / commutative). Subscript/스칼라 섞임/깊은 체인/mul-sub 는
  개별 연산자로 폴백 (Phase 2 의 47x 가속은 유지).
- **추론 데모는 1샘플** — 배치 inference (2D×2D matmul) 미구현.
- **matmul 은 naive** — tiling/blocking 없는 단일 직선 루프. auto-vectorize 에
  맡김. BLAS 급 성능 아님.
- **Subview/ownership** (Nadya 시그니처 기능)은 미구현.
- **C 확장은 Python 버전별 재빌드** 가 필요하다.
- 출력은 `ast.unparse` 로 생성되므로 원본의 빈 줄·주석 포매팅은 보존되지
  않는다.

## 테스트 / 데모 / 벤치마크

```bash
# 전체 테스트 (Phase 2 이상이면 C 확장 빌드 후)
python setup.py build_ext --inplace
python -m pytest

# 예시 (각각 PYTHONPATH=. 필요)
python examples/readme_example.py        # 부분평가
python examples/decorator_example.py     # @specialize
python examples/import_example.py        # import 시점 특수화
python examples/parallel_example.py      # attr[parallel]
python examples/unroll_example.py        # attr[unroll]
python examples/tensor_example.py        # C 텐서
python examples/fusion_example.py        # 표현식 융합
python examples/xor_inference.py         # XOR MLP 추론
python examples/digit_inference.py       # 손글씨 숫자 8×8 MLP 추론

# 기능별 벤치마크 (각자 그 기능이 돋보이는 워크로드)
python benchmarks/bench_tensor_ops.py       # C Tensor raw ops
python benchmarks/bench_madd.py             # 표현식 융합 madd
python benchmarks/bench_linear_relu.py      # Dense+ReLU 융합
python benchmarks/bench_unroll.py           # attr[unroll]
python benchmarks/bench_parallel.py         # attr[parallel]

# 종합 추론 벤치 (4단계 분해)
python benchmarks/feature_breakdown_benchmark.py        # small/medium/large/huge
python benchmarks/digit_inference_benchmark.py          # PRE vs COMPILED

# 초기 벤치 (개발 단계용, 참고)
python benchmarks/parallel_benchmark.py
python benchmarks/tensor_benchmark.py
python benchmarks/fusion_benchmark.py
python benchmarks/inference_benchmark.py
```

### 오프라인 학습 (한 번)

`examples/digit_weights.json` 는 `tools/train_digits.py` 가 sklearn 으로
한 번 학습해 박아 둔 결과다. 재생성하려면:

```bash
pip install numpy scikit-learn   # 학습용, pydya 실행엔 불필요
python tools/train_digits.py
```

학습은 pydya 의 일부가 아니다 — Optimium 처럼 *학습된 가중치를 받아 추론*
하는 컴파일러의 위치를 그대로 따른다.

## 참고

- [Optimium 시리즈 by ENERZAi](https://medium.com/@enerzai) — Nadya 의 설계
  배경 (`template<>`, `attr[Parallel]`, MLIR 기반 lowering, FMA, ownership 등)
