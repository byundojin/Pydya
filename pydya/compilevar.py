"""Pydya의 컴파일 타임 변수 마커."""


class CompileVar:
    """컴파일 시점에 값이 고정되는 심볼.

    두 가지 선언 형태를 지원한다.

    * 타입형(정식): ``V = CompileVar[int]()`` — 레이블은 대입 대상 변수
      이름(``V``)에서 추론된다. ``[int]`` 은 컴파일 타임 값의 타입 힌트로
      현재는 문서화 용도이며 실행/검증에 영향을 주지 않는다.
    * 문자열형: ``V = CompileVar('label')`` — 레이블을 명시적으로 지정한다.

    실제 값은 ``env`` 매핑에서 레이블 키로 :func:`pydya.compile_source`
    (또는 데코레이터/임포트 경로)에 전달된다.
    """

    def __init__(self, name: str | None = None):
        self.name = name

    def __class_getitem__(cls, item):
        # ``CompileVar[int]`` 형태를 호출 가능하게 만든다. 타입 인자는
        # 런타임에는 무시되고, 선언 레이블/검증은 소스 AST 에서 처리된다.
        return cls

    def __repr__(self) -> str:
        return f"CompileVar({self.name!r})"
