# Pydya

Pydya 는 Python 소스를 **부분 평가(partial evaluation)** 하는 소스-투-소스
컴파일러다. 일부 값이 컴파일 시점에 고정되어 있다고 알려 주면, 그 값에 의존하는
계산을 미리 수행하고 결정된 분기·함수 호출을 펼쳐, 더 단순하고 빠른 잔여
(residual) 프로그램을 만들어 낸다.

이름이 시사하듯 설계는 Nadya 스타일의 2단계(정적/동적) 분리에서
영감을 받았다. 정적으로 알려진 값은 미리 계산하고, 나머지는 런타임 코드로
그대로 남긴다.

## 핵심 아이디어

컴파일 시점에 고정하고 싶은 값을 `CompileVar` 로 선언한다.

```python
from pydya import CompileVar, compile_source

source = '''
V = CompileVar('V')

def f(a):
    return a + V

if V < 5:
    a = f(5)
else:
    a = 5

b = a + V

print(a)
print(b)
'''

print(compile_source(source, env={"V": 3}))
```

`env={"V": 3}` 으로 컴파일하면 다음이 출력된다.

```python
def f(a):
    return a + 3
a = 8
b = a + 3
print(a)
print(b)
```

무슨 일이 일어났는가:

- `V` 는 `3` 으로 알려져 있으므로 모든 `V` 사용처가 `3` 으로 치환된다.
  함수 `f` 도 `return a + 3` 으로 특수화된다.
- `if V < 5` 는 `3 < 5` → `True` 로 접혀, `else` 가지가 통째로 제거된다.
- `a = f(5)` 는 상수 인자 호출이므로 인라인되어 `5 + 3` → `8` 로 계산된다.
- `b = a + V` 에서 `V` 만 `3` 으로 접힌다. `a` 는 런타임 바인딩이므로
  `a = 8` 을 알고 있더라도 그 값을 전파하지 않고 `b = a + 3` 으로 남긴다.

`env={"V": 9}` 로 컴파일하면 `V < 5` 가 거짓이 되어 `a = 5` 가지가 선택되고
`f` 는 호출되지 않는다.

## 설치

```bash
pip install -e .
```

Python 3.9 이상이 필요하다. 런타임 의존성은 없다(표준 라이브러리 `ast` 만 사용).

## API

### `CompileVar(name)`

컴파일 시점에 값이 고정되는 심볼을 선언하는 마커. `name` 은 `env` 에서 값을
찾을 때 쓰는 레이블이며, 변수 이름과 달라도 된다.

```python
flag = CompileVar('debug')   # env={'debug': True}
```

### `compile_source(source, env=None)`

`source`(문자열)를 `env` 매핑을 기준으로 부분 평가하여 변환된 소스를 문자열로
반환한다. `env` 는 `CompileVar` 의 레이블을 컴파일 타임 값으로 매핑한다.
선언된 `CompileVar` 의 값이 `env` 에 없으면 `MissingCompileValue` 가 발생한다.

## 동작 원리: 패스 파이프라인

`compile_source` 는 소스를 AST 로 파싱한 뒤 다음 패스를 순서대로 적용하고
다시 소스로 출력한다.

| 순서 | 패스 | 모듈 | 역할 |
|------|------|------|------|
| 1 | collect | `passes/collect.py` | `CompileVar` 선언과 `pydya` import 를 제거하고 정적 환경을 만든다 |
| 2 | fold | `passes/fold.py` | 정적 이름을 치환하고 상수 산술·비교·불리언·단항식을 폴딩한다 |
| 3 | branch | `passes/branch.py` | 상수로 접힌 `if`/`while`/조건식의 선택된 가지만 남긴다 |
| 4 | inline | `passes/inline.py` | 상수 인자로 호출되는 단순 함수를 인라인·특수화한다 |
| 5 | dce | `passes/dce.py` | 함수 본문 내 부작용 없는 죽은 대입을 고정점까지 제거한다 |

## 설계상 결정과 한계

부분 평가는 **관찰 가능한 동작을 바꾸지 않을 때만** 변환을 수행해야 한다.
Pydya 는 안전을 위해 보수적으로 동작한다.

- **정적 이름만 치환한다.** 일반 변수의 우변이 상수로 계산되더라도 그 변수의
  사용처를 전파하지 않는다(`a = 8` 이어도 `b = a + 3`).
- **상수 인자 호출만 인라인한다.** 런타임 식을 인라인하면 부작용이 중복되거나
  실행 순서가 바뀔 수 있으므로 하지 않는다. 또한 단일 `return` 본문에 위치
  인자만 받는 함수만 대상으로 한다.
- **모듈 최상위 바인딩은 제거하지 않는다.** 다른 모듈에서 import 될 수 있어
  이 소스만으로 미사용임을 증명할 수 없기 때문이다.
- 출력은 `ast.unparse` 로 생성되므로 원본의 빈 줄·주석 등 포매팅은 보존되지
  않는다.

향후 확장 여지: 정적 값에 대한 루프 펼치기, `list`/`dict` 등 컨테이너 상수
폴딩, 다중 문장 함수 인라인, CLI.

## 테스트

```bash
PYTHONPATH=. python -m pytest
```

각 패스별 단위 테스트(`tests/test_*.py`)와 전체 파이프라인을 검증하는
엔드 투 엔드 테스트(`tests/test_e2e.py`)가 있다. 엔드 투 엔드 테스트는 잔여
프로그램을 실제로 실행해 원본과 같은 값을 출력하는지까지 확인한다.

## 데모

```bash
PYTHONPATH=. python examples/readme_example.py
```
