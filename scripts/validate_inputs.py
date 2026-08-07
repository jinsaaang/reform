"""Validate the portable benchmark files required by canonical HGF."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "hgf"


def main() -> int:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SOURCE_ROOT), *([existing] if existing else [])]
    )
    command = [sys.executable, "-m", "hgf.validate", *sys.argv[1:]]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
