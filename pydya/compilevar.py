"""Compile-time variable marker for Pydya."""


class CompileVar:
    """A symbol whose value is fixed at compile time.

    Declaring ``X = CompileVar('name')`` in source marks ``X`` as a static
    binding. The concrete value is supplied to :func:`pydya.compile_source`
    through the ``env`` mapping keyed by ``name``.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"CompileVar({self.name!r})"
