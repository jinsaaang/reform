"""Verify the frozen bundle plus additive experiment-extension files.

The original artifact manifest remains untouched. Its listed files are checked
exactly, while additive experiment files are hashed as a separate live section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_common import (  # noqa: E402
    _live_extension_digest,
    read_json,
)
from hgf.manifest import _file_record  # noqa: E402
from hgf.preflight import run_preflight  # noqa: E402


def _frozen_manifest_errors() -> list[str]:
    manifest = read_json(ROOT / "artifact_manifest.json")
    errors = []
    for relative, expected in manifest.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing frozen file: {relative}")
            continue
        actual = _file_record(path)
        if actual != expected:
            errors.append(f"frozen hash or size mismatch: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-evidence-content",
        action="store_true",
        help="Read all 200 SQLite banks in addition to coverage checks.",
    )
    args = parser.parse_args()
    errors = _frozen_manifest_errors()
    preflight = run_preflight(
        validate_evidence=args.check_evidence_content
    )
    errors.extend(preflight["errors"])
    extension = _live_extension_digest(ROOT)
    if not extension["file_count"]:
        errors.append("no experiment extension files found")
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "frozen_manifest_policy": (
            "Verify every originally listed file; permit additive experiment "
            "files without rewriting the frozen manifest."
        ),
        "bundle_preflight": preflight,
        "experiment_extensions": extension,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
