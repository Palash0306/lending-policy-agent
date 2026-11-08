"""The web page's behaviour (asking, batches, reload, downloads, safe text), run in a small simulated browser with node.
Skipped when node is not installed. The scenarios live in tests/ui/."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent / "ui"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed (it is only needed for the web page tests)")
def test_every_web_page_scenario_passes():
    done = subprocess.run(["node", str(UI / "run_all.js")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ALL PASSED" in done.stdout
