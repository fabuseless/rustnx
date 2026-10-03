import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_api_docs_are_up_to_date():
    # docs/API.md is generated from the code; regenerate it with
    # `python scripts/gen_api_docs.py` after adding or changing a function.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_api_docs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
