"""Nadya 스타일 병렬 어트리뷰트 마커.

Nadya 의 ``attr[Parallel : true] for(...)`` 에 대응한다. 소스에서 for 루프
바로 앞에 ::

    from pydya import attr

    attr[{'parallel': True}]
    for i in range(n):
        out[i] = f(i)

처럼 두면, Pydya 컴파일러가 그 다음 for 루프를 병렬 실행 코드로 변환한다.
컴파일되지 않은 런타임에서 ``attr[...]`` 자체는 아무 동작도 하지 않는다.
"""


class _Attr:
    def __getitem__(self, options):
        return None

    def __repr__(self) -> str:
        return "attr"


attr = _Attr()
