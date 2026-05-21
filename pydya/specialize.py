"""``@specialize`` 데코레이터: 컴파일 타임 지역변수를 바인딩해 함수를 특수화한다.

사용 형태::

    @specialize({'V': 3})
    def f(a):
        V = CompileVar[int]()
        return a + V

데코레이터는 함수 소스를 가져와 본문의 ``CompileVar`` 선언을 ``env`` 값으로
바인딩하고, 폴딩/분기제거/인라인/DCE 파이프라인을 적용한 뒤, 원본 함수의
전역(``__globals__``)에서 다시 컴파일한 특수화 함수를 돌려준다. 특수화된
소스 문자열은 ``__pydya_source__`` 속성으로 확인할 수 있다.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import Any, Callable, Mapping

from pydya.compiler import optimize
from pydya.passes.collect import collect_in_body


def specialize(env: Mapping[str, Any]) -> Callable[[Callable], Callable]:
    """주어진 컴파일 타임 환경으로 함수를 특수화하는 데코레이터를 만든다."""
    env = dict(env)

    def decorator(fn: Callable) -> Callable:
        if getattr(fn, "__closure__", None):
            raise ValueError(
                f"specialize: {fn.__name__!r} 가 자유변수 "
                f"{fn.__code__.co_freevars} 를 캡처합니다. 클로저는 아직 지원되지 않습니다."
            )

        source = textwrap.dedent(inspect.getsource(fn))
        module = ast.parse(source)
        funcdef = module.body[0]
        if not isinstance(funcdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError("specialize 는 함수에만 적용할 수 있습니다.")

        # 재실행 시 데코레이터가 다시 트리거되지 않도록 제거한다.
        funcdef.decorator_list = []

        static_values, funcdef.body = collect_in_body(funcdef.body, env)
        optimize(funcdef, static_values)
        if not funcdef.body:
            funcdef.body = [ast.Pass()]
        ast.fix_missing_locations(module)

        specialized_source = ast.unparse(funcdef)
        namespace: dict = {}
        code = compile(module, filename=f"<specialize:{fn.__name__}>", mode="exec")
        exec(code, fn.__globals__, namespace)

        new_fn = namespace[fn.__name__]
        functools.update_wrapper(new_fn, fn)
        new_fn.__pydya_source__ = specialized_source
        return new_fn

    return decorator
