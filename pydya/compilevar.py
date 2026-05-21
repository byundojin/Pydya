"""Pydya의 컴파일 타임 변수 마커."""


class CompileVar:
    """컴파일 시점에 값이 고정되는 심볼.

    소스에서 ``X = CompileVar('name')`` 으로 선언하면 ``X`` 는 정적(static)
    바인딩으로 표시된다. 실제 값은 ``env`` 매핑에서 ``name`` 키로
    :func:`pydya.compile_source` 에 전달된다.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"CompileVar({self.name!r})"
