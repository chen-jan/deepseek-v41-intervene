"""(a) DeepSeek's own test_encoding.py passes in this venv, and the provider loads that exact encoding.py.

Two of the official tests read the checkpoint's ``inference/examples``. When ``$DSV41_OFFICIAL_DIR`` (or
``$DSV41_MODEL_DIR``) holds them, the full suite runs on a byte-identical mirror in a temp dir, so nothing is ever
written into the checkpoint copy; otherwise that test is skipped.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import OFFICIAL_DIR, dsp

ENC_DIR = OFFICIAL_DIR / "encoding"
EXAMPLES = OFFICIAL_DIR / "inference" / "examples"
NEEDS_EXAMPLES = ("test_examples_vl_txt_and_json_encode_identically", "test_examples_harmony_cases_encode")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run_pytest(test_file: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file), *extra],
                          cwd=test_file.parent, env=env, capture_output=True, text=True, timeout=300)


def _passed(out: str) -> int:
    m = re.search(r"(\d+) passed", out)
    return int(m.group(1)) if m else 0


def test_provider_loads_the_checkpoint_encoding(enc):
    assert Path(enc.__dsv41_path__) == (ENC_DIR / "encoding.py").resolve()
    assert enc.__dsv41_sha256__ == dsp.OFFICIAL_ENCODING_SHA256 == _sha(ENC_DIR / "encoding.py")


def test_official_suite_in_place():
    """In place (read-only use of the checkpoint copy): everything except the two tests needing inference/examples."""
    before = sorted(p.name for p in ENC_DIR.iterdir())
    deselect = [f"--deselect=test_encoding.py::{t}" for t in NEEDS_EXAMPLES]
    r = _run_pytest(ENC_DIR / "test_encoding.py", *deselect)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert _passed(r.stdout) == 53, r.stdout[-500:]
    assert sorted(p.name for p in ENC_DIR.iterdir()) == before, "the official test run wrote into the checkpoint copy"


def test_official_suite_full_with_checkpoint_examples(tmp_path):
    """All 55 official tests, on a byte-identical mirror plus the checkpoint's inference/examples."""
    if not EXAMPLES.is_dir():
        pytest.skip(f"no {EXAMPLES} (the checkpoint's inference/examples)")
    mirror = tmp_path / "encoding"
    shutil.copytree(ENC_DIR, mirror, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(EXAMPLES, tmp_path / "inference" / "examples")
    for f in ["encoding.py", "test_encoding.py", "README.md", *[f"tests/{p.name}" for p in (ENC_DIR / "tests").iterdir()]]:
        assert _sha(mirror / f) == _sha(ENC_DIR / f), f
    r = _run_pytest(mirror / "test_encoding.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert _passed(r.stdout) == 55, r.stdout[-500:]
