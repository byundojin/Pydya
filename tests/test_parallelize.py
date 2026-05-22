import ast

import pytest

from pydya import compile_source
from pydya.passes.parallelize import UnsafeParallelLoop, parallelize


def _run(src):
    compiled = compile_source(src)
    ns = {}
    exec(compiled, ns)
    return compiled, ns


def test_independent_map_transforms_and_runs():
    compiled, ns = _run(
        "from pydya import attr\n"
        "out = [0] * 5\n"
        "attr[{'parallel': True}]\n"
        "for i in range(5):\n"
        "    out[i] = i * i\n"
    )
    assert "parallel_map_into" in compiled
    assert "pydya.runtime" in compiled
    assert ns["out"] == [0, 1, 4, 9, 16]


def test_workers_option_passed_through():
    compiled, ns = _run(
        "from pydya import attr\n"
        "out = [0] * 4\n"
        "attr[{'parallel': True, 'workers': 2}]\n"
        "for i in range(4):\n"
        "    out[i] = i + 1\n"
    )
    assert "workers=2" in compiled
    assert ns["out"] == [1, 2, 3, 4]


def test_reads_outer_captures():
    compiled, ns = _run(
        "from pydya import attr\n"
        "factor = 10\n"
        "out = [0] * 3\n"
        "attr[{'parallel': True}]\n"
        "for i in range(3):\n"
        "    out[i] = i * factor\n"
    )
    assert ns["out"] == [0, 10, 20]


def test_parallel_false_leaves_serial_loop():
    compiled, ns = _run(
        "from pydya import attr\n"
        "out = [0] * 3\n"
        "attr[{'parallel': False}]\n"
        "for i in range(3):\n"
        "    out[i] = i\n"
    )
    assert "parallel_map_into" not in compiled
    assert "attr" not in compiled  # 마커는 제거된다
    assert ns["out"] == [0, 1, 2]


def test_destructive_update_rejected():
    # 누적(append)은 반복 간 공유 상태 갱신 → 거부
    src = (
        "from pydya import attr\n"
        "out = []\n"
        "attr[{'parallel': True}]\n"
        "for i in range(3):\n"
        "    out.append(i)\n"
    )
    with pytest.raises(UnsafeParallelLoop):
        compile_source(src)


def test_reading_target_rejected():
    # expr 가 out 을 읽으면 반복 간 의존 → 거부
    src = (
        "from pydya import attr\n"
        "out = [1] * 3\n"
        "attr[{'parallel': True}]\n"
        "for i in range(3):\n"
        "    out[i] = out[i - 1] + 1\n"
    )
    with pytest.raises(UnsafeParallelLoop):
        compile_source(src)


def test_marker_without_for_rejected():
    src = "from pydya import attr\nattr[{'parallel': True}]\nx = 1\n"
    with pytest.raises(UnsafeParallelLoop):
        compile_source(src)


def test_parallelize_is_idempotent_on_plain_code():
    tree = ast.parse("a = 1\nfor i in range(3):\n    a = i\n")
    parallelize(tree)
    assert "parallel_map_into" not in ast.unparse(tree)
