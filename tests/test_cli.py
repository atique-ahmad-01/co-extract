import json
import os
import subprocess
import sys

from tests.conftest import ROOT, SYNTH


def run_cli(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER")}
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "co_extract", *args], cwd=ROOT / "tests",
                          capture_output=True, text=True, env=env, timeout=60)


def test_cli_offline_prints_json():
    res = run_cli(str(SYNTH / "co_02_credit_pending.txt"), "--offline")
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["amount"]["value"] == -10630.0
    assert out["meta"]["mode"] == "offline"


def test_cli_without_key_fails_loudly():
    res = run_cli(str(SYNTH / "co_01_aia_style.txt"))
    assert res.returncode == 2
    assert "LLM_API_KEY" in res.stderr
