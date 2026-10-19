"""P4-5: `ve port` mode — version x hardware matrix plumbing (local_smoke). No GPU."""
from __future__ import annotations

import json

from vllm_evolve.cli import main as ve_cli
from vllm_evolve.tools import phase_guard


def test_port_local_smoke_matrix_plumbing(tmp_path, capsys):
    phase_guard.reset_state()
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    out = tmp_path / "port_report.json"
    rc = ve_cli.main(["port", str(pol), "scheduling", "--versions", "0.21.0,0.22.0",
                      "--hardware", "h100", "--backend", "local_smoke", "--out", str(out)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and res["ok"] is True and res["mode"] == "port"
    assert len(res["matrix"]) == 2                       # 2 versions x 1 hardware
    # every off-box cell is synthetic and non-promotable (never real_vllm -> never a gain)
    for cell in res["matrix"]:
        assert cell["source"] != "real_vllm"
        assert cell["hardware_id"] == "h100"
    assert res["summary"]["real_cells"] == 0
    assert out.is_file() and json.loads(out.read_text(encoding="utf-8"))["mode"] == "port"
    phase_guard.reset_state()


def test_port_rejects_non_scheduling_and_missing_policy(tmp_path, capsys):
    phase_guard.reset_state()
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    rc = ve_cli.main(["port", str(pol), "kv_eviction", "--versions", "0.21.0",
                      "--hardware", "h100", "--backend", "local_smoke"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "outcome_class"] == "unsupported_target"
    rc = ve_cli.main(["port", str(tmp_path / "nope.py"), "--versions", "0.21.0",
                      "--hardware", "h100", "--backend", "local_smoke"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "outcome_class"] == "policy_not_found"
    phase_guard.reset_state()
