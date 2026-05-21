"""import 시점 특수화 opt-in 마커.

모듈 상단에서 ::

    from pydya.module import specialize_here

처럼 import 하면, pydya import 훅(:func:`pydya.importer.install`)이 설치된
상태에서 그 모듈은 로드 시 :func:`pydya.compile_source` 로 부분 평가된다.
``from __future__ import annotations`` 와 같은 모듈 단위 디렉티브 역할이며,
``specialize_here`` 자체에는 런타임 의미가 없다(훅이 소스에서 이 마커의
존재만 확인한다).
"""

MARKER = "specialize_here"

# 마커 import 가 (훅 미설치 시에도) 실패하지 않도록 하는 더미 심볼.
specialize_here = object()
