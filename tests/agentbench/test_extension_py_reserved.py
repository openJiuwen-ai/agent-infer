# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Reserve the ``extension.py`` filename for the JiuwenSwarm integration.

JiuwenSwarm's extension loader scans the immediate child directories of each
``extension_dirs`` entry for a file named exactly ``extension.py`` and imports
it at Gateway startup (see jiuwenswarm/extensions/loader.py). AgentBench sets
``extension_dirs`` to the ``agents/`` directory, so any
``agents/<runtime>/extension.py`` would be auto-imported as a Jiuwen extension
on every Jiuwen run — a trap a future maintainer can hit by simply naming a
new integration file. This test reserves the name for the JiuwenSwarm
integration until upstream supports specifying an exact extension root.
"""

from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parents[2] / "agentinfer" / "agentbench" / "agents"


def test_extension_py_is_reserved_for_jiuwenswarm() -> None:
    """Only ``jiuwenswarm/extension.py`` may use that filename under agents/."""

    found = {str(path.relative_to(AGENTS_DIR).as_posix()) for path in AGENTS_DIR.rglob("extension.py")}
    assert found == {"jiuwenswarm/extension.py"}, (
        "agents/*/extension.py is auto-imported by Jiuwen's loader; reserve it "
        f"for the JiuwenSwarm integration. Found: {sorted(found)}"
    )
