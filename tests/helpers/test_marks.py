# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pin registry validation and collection behavior used by CI selection."""

from __future__ import annotations

import ast
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.marks import resolve_pytest_marks, test_marks, validate_marks

pytest_plugins = ["pytester"]
pytestmark = test_marks(hardware="cpu", ci="ready")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("fields", "names"),
    [
        ({"hardware": "cpu"}, ["cpu"]),
        ({"hardware": "cpu", "ci": "ready"}, ["cpu", "ready"]),
        ({"hardware": "L20", "num_cards": 2}, ["L20", "cards_2"]),
        ({"hardware": "A3", "num_cards": 16, "ci": "nightly"}, ["A3", "cards_16", "nightly"]),
    ],
)
def test_valid_declarations(pytestconfig, fields, names):
    marks = resolve_pytest_marks(fields, pytestconfig)
    assert [mark.name for mark in marks] == names


@pytest.mark.parametrize(
    "fields",
    [
        None,
        [],
        {},
        {"hardware": "cpu", "num_cards": 1},
        {"hardware": "L20"},
        {"hardware": "L20", "num_cards": True},
        {"hardware": "L20", "num_cards": 0},
        {"hardware": "L20", "num_cards": 8},
        {"hardware": "Unknown", "num_cards": 1},
        {"hardware": "cpu", "ci": "unknown"},
        {"hardware": "cpu", "extra": 1},
    ],
)
def test_invalid_declarations(pytestconfig, fields):
    unknown = isinstance(fields, dict) and (fields.get("num_cards") == 8 or fields.get("hardware") == "Unknown")
    with pytest.warns(pytest.PytestUnknownMarkWarning) if unknown else nullcontext():
        with pytest.raises(ValueError):
            resolve_pytest_marks(fields, pytestconfig)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["cpu", "L20"], "multiple model"),
        (["L20", "cards_1", "cards_2"], "multiple cards"),
        (["cpu", "ready", "nightly"], "multiple lane"),
        (["cpu", "cpu"], "multiple model"),
        (["ready"], "missing model"),
        (["L20"], "declare num_cards"),
        (["cpu", "cards_1"], "must not specify"),
        (["cpu", "unregistered"], "unregistered marker"),
    ],
)
def test_invalid_marker_sets(pytestconfig, names, message):
    marks = [SimpleNamespace(name=name) for name in names]
    with pytest.raises(ValueError, match=message):
        validate_marks(marks, pytestconfig)


@pytest.fixture
def collection_tree(pytester):
    # Run the production hook in a fresh pytest process without importing vLLM.
    for relative in (
        "pyproject.toml",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/helpers/__init__.py",
        "tests/helpers/marks.py",
        "tests/e2e/__init__.py",
        "tests/e2e/helpers/__init__.py",
        "tests/e2e/helpers/case_loader.py",
    ):
        target = pytester.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return pytester


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("[pytest.mark.cpu, pytest.mark.cards_8]", "unregistered marker"),
        ("[pytest.mark.cpu, pytest.mark.L20]", "multiple model"),
        ("[pytest.mark.ready]", "missing model"),
    ],
)
def test_collection_rejects_invalid_marks(collection_tree, declaration, message):
    path = collection_tree.path / "tests/test_sample.py"
    path.write_text(f"import pytest\npytestmark = {declaration}\ndef test_sample(): pass\n", encoding="utf-8")
    result = collection_tree.runpytest_subprocess("tests", "--collect-only", "-q", "-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert message in result.stderr.str()


def test_core_declarations_excluded_from_ready(collection_tree):
    core = collection_tree.path / "tests/agentcache/core"
    core.mkdir(parents=True)
    sources = sorted((ROOT / "tests/agentcache/core").glob("test_*.py"))
    assert sources
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        declarations = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
        ]
        assert len(declarations) == 1, source
        # Exercise the actual declaration independently of optional vLLM imports.
        declaration = ast.unparse(declarations[0])
        (core / source.name).write_text(
            "from tests.helpers.marks import test_marks\n" + declaration + "\ndef test_core(): pass\n",
            encoding="utf-8",
        )
    (collection_tree.path / "tests/test_ready.py").write_text(
        'from tests.helpers.marks import test_marks\npytestmark = test_marks(hardware="cpu", ci="ready")\n'
        "def test_ready(): pass\n",
        encoding="utf-8",
    )
    result = collection_tree.runpytest_subprocess("tests", "-q", "-m", "cpu and ready", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, deselected=len(sources))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"hardware": "cpu"}, None),
        ({"hardware": "L20", "num_cards": 2}, None),
        ({"hardware": "L20", "num_cards": 2, "timeout": 180}, None),
        ({"hardware": "cpu", "timeout": -1}, "finite positive"),
        ({"hardware": "L20", "num_cards": 8}, "unregistered marker"),
        ({"ci": "ready"}, "mark requires hardware"),
        (["cpu"], "mark must be an object"),
    ],
)
def test_e2e_json_collection(collection_tree, fields, message):
    import json

    suite = collection_tree.path / "tests/e2e"
    (suite / "cases").mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/e2e/conftest.py", suite / "conftest.py")
    (suite / "cases/sample.json").write_text(json.dumps({"mark": fields}), encoding="utf-8")
    (suite / "test_sample.py").write_text("def test_sample(case_file): pass\n", encoding="utf-8")
    result = collection_tree.runpytest_subprocess("tests/e2e", "--collect-only", "-q", "-p", "no:cacheprovider")
    if message is None:
        assert result.ret == pytest.ExitCode.OK
        assert "test_sample[sample]" in result.stdout.str()
    else:
        assert result.ret != pytest.ExitCode.OK
        assert message in result.stdout.str() + result.stderr.str()


def test_e2e_benchmark_array_expands_with_marks_and_filtering(collection_tree):
    """One selected JSON expands into independently selectable benchmark entries."""
    import json

    suite = collection_tree.path / "tests/e2e"
    (suite / "cases").mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/e2e/conftest.py", suite / "conftest.py")
    payload = {
        "mark": {"hardware": "L20", "num_cards": 2, "ci": "ready"},
        "benchmark_params": [
            {"agent-type": "claude", "agent-profile": "single"},
            {"agent-type": "dsh", "agent-profile": "single"},
            {"agent-type": "zcode", "agent-profile": "plan-subagent"},
        ],
    }
    (suite / "cases/matrix.json").write_text(json.dumps(payload), encoding="utf-8")
    (suite / "test_sample.py").write_text(
        "def test_sample(case_file, benchmark_index):\n"
        "    assert case_file.name == 'matrix.json'\n"
        "    assert benchmark_index in (0, 1, 2)\n",
        encoding="utf-8",
    )
    result = collection_tree.runpytest_subprocess(
        "tests/e2e", "-q", "-m", "L20 and cards_2 and ready", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=3)
    result = collection_tree.runpytest_subprocess(
        "tests/e2e",
        "--test-config-file",
        "tests/e2e/cases/matrix.json",
        "-q",
        "-k",
        "zcode and plan",
        "-p",
        "no:cacheprovider",
    )
    result.assert_outcomes(passed=1, deselected=2)


@pytest.mark.parametrize("mode", ["reuse", "startup-failure", "different-deployment"])
def test_agentbench_session_owns_one_model_process(collection_tree, mode):
    """Exercise the real fixture's shared lifetime and cleanup in a pytest process."""
    import json

    e2e = collection_tree.path / "tests/e2e"
    suite = e2e / "agentbench"
    (suite / "cases").mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/e2e/conftest.py", e2e / "conftest.py")
    shutil.copyfile(ROOT / "tests/e2e/helpers/server.py", e2e / "helpers/server.py")
    payload = json.loads(
        (ROOT / "tests/e2e/function/agentbench/cases/qwen3_coder_agents_1x1.json").read_text(encoding="utf-8")
    )
    payload["mark"] = {"hardware": "cpu", "ci": "ready"}
    (suite / "cases/matrix.json").write_text(json.dumps(payload), encoding="utf-8")
    fixture_source = (ROOT / "tests/e2e/function/agentbench/conftest.py").read_text(encoding="utf-8")
    fixture_source += (
        "\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "class FakeVllmServer:\n"
        "    def __init__(self, *, config):\n"
        "        self.config = config\n"
        "        self.proc = None\n"
        "    def start(self):\n"
        "        with Path('server-events.txt').open('a') as f: f.write('start\\n')\n"
        f"        if {mode == 'startup-failure'!r}: raise RuntimeError('model startup failed')\n"
        "        Path('server-port.txt').write_text(str(self.config.port))\n"
        "        self.proc = SimpleNamespace(poll=lambda: None)\n"
        "    def stop(self):\n"
        "        with Path('server-events.txt').open('a') as f: f.write('stop\\n')\n"
        "VllmServer = FakeVllmServer\n"
    )
    (suite / "conftest.py").write_text(fixture_source, encoding="utf-8")
    (suite / "test_shared.py").write_text(
        "from pathlib import Path\nfrom dataclasses import replace\n"
        "from tests.e2e.helpers.case_loader import load_pytest_case\n"
        "def test_shared(pytestconfig, case_file, benchmark_index, agentbench_server):\n"
        "    config = load_pytest_case(pytestconfig, case_file, benchmark_index=benchmark_index)\n"
        f"    if {mode == 'different-deployment'!r} and benchmark_index == 1:\n"
        "        config = replace(config, host='localhost')\n"
        "    bound = agentbench_server(config)\n"
        "    assert bound.port == int(Path('server-port.txt').read_text())\n"
        "    assert bound.case.benchmark_params == config.case.benchmark_params\n",
        encoding="utf-8",
    )
    args = ["tests/e2e/agentbench", "-q", "-p", "no:cacheprovider"]
    if mode == "startup-failure":
        args.extend(["-k", "claude and single"])
    result = collection_tree.runpytest_subprocess(*args)
    if mode == "reuse":
        result.assert_outcomes(passed=7)
    elif mode == "startup-failure":
        result.assert_outcomes(failed=1, deselected=6)
        assert "model startup failed" in result.stdout.str()
    else:
        result.assert_outcomes(passed=6, failed=1)
        assert "must share the same deployment" in result.stdout.str()
    assert (collection_tree.path / "server-events.txt").read_text().splitlines() == ["start", "stop"]


@pytest.mark.parametrize("seconds", [180, 0.5])
def test_optional_json_timeout(pytestconfig, seconds):
    fields = {"hardware": "L20", "num_cards": 2, "ci": "ready", "timeout": seconds}
    marks = resolve_pytest_marks(fields, pytestconfig)
    assert [mark.name for mark in marks] == ["L20", "cards_2", "ready", "timeout"]
    assert marks[-1].args == (seconds,)
    python_marks = test_marks(hardware="L20", num_cards=2, ci="ready", timeout=seconds)
    assert [(mark.name, mark.args) for mark in python_marks] == [(mark.name, mark.args) for mark in marks]
    assert fields["timeout"] == seconds
    assert all(mark.name != "timeout" for mark in resolve_pytest_marks({"hardware": "cpu"}, pytestconfig))


@pytest.mark.parametrize("seconds", [None, True, "180", 0, -1, float("nan"), float("inf")])
def test_invalid_json_timeout(pytestconfig, seconds):
    with pytest.raises(ValueError, match="finite positive"):
        resolve_pytest_marks({"hardware": "cpu", "timeout": seconds}, pytestconfig)


def test_json_timeout_requires_plugin(pytestconfig, monkeypatch):
    monkeypatch.setattr(pytestconfig.pluginmanager, "hasplugin", lambda name: False)
    with pytest.raises(ValueError, match="requires pytest-timeout"):
        resolve_pytest_marks({"hardware": "cpu", "timeout": 180}, pytestconfig)


@pytest.mark.parametrize("seconds", [True, "180", 0, -1, float("nan"), float("inf")])
def test_invalid_python_timeout(seconds):
    with pytest.raises(ValueError, match="finite positive"):
        test_marks(hardware="cpu", timeout=seconds)


def test_python_timeout_omitted():
    assert [mark.name for mark in test_marks(hardware="cpu")] == ["cpu"]
    assert [mark.name for mark in test_marks(hardware="cpu", timeout=None)] == ["cpu"]
