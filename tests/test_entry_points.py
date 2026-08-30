"""The scripts in `bin/` are the only public interface; check they still start.

Each one is run as a subprocess from a directory that is not the repository
root, which is what catches an import or a default path that only resolved
because of the cwd.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hidden_eclipse.paths import DEFAULT_GLOBE, DEFAULT_POLICY, ROOT


SCRIPTS = ["globe.py", "train.py", "evaluate.py", "serve.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_from_anywhere(script: str, tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "bin" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_artifact_defaults_point_at_files_that_exist() -> None:
    assert DEFAULT_POLICY.is_file()
    assert DEFAULT_GLOBE.is_file()


def test_globe_writes_a_page_without_opening_a_browser(tmp_path) -> None:
    out = tmp_path / "globe.html"
    result = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "globe.py"), "--seed", "1337", "--no-open",
         "--output", str(out)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert "<html" in out.read_text(encoding="utf-8").lower()
