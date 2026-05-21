import importlib
import sys

import pydya.importer as importer
from pydya import CompileVar


def _write(tmp_path, name, body):
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_marked_module_is_specialized(tmp_path, monkeypatch):
    _write(
        tmp_path,
        "kernel_mod",
        "from pydya import CompileVar\n"
        "from pydya.importer import specialize_here\n"
        "SCALE = CompileVar[int]()\n"
        "def scaled(x):\n"
        "    return x * SCALE\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importer.reset()
    importer.install({"SCALE": 4})
    sys.modules.pop("kernel_mod", None)
    try:
        mod = importlib.import_module("kernel_mod")
        assert mod.scaled(10) == 40
        assert not hasattr(mod, "SCALE")
    finally:
        importer.reset()
        sys.modules.pop("kernel_mod", None)


def test_unmarked_module_untouched(tmp_path, monkeypatch):
    _write(
        tmp_path,
        "plain_mod",
        "from pydya import CompileVar\n"
        "SCALE = CompileVar[int]()\n"
        "value = SCALE\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importer.reset()
    importer.install({"SCALE": 4})
    sys.modules.pop("plain_mod", None)
    try:
        mod = importlib.import_module("plain_mod")
        # 마커가 없으므로 특수화되지 않고 CompileVar 인스턴스가 그대로 남는다.
        assert isinstance(mod.SCALE, CompileVar)
    finally:
        importer.reset()
        sys.modules.pop("plain_mod", None)
