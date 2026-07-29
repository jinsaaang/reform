from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .package import PACKAGE_ROOT


MANIFEST_PATH = PACKAGE_ROOT / "artifact_manifest.json"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    "runs",
}
EXCLUDED_FILES = {
    ".env",
    "artifact_manifest.json",
}


def _included_files() -> list[Path]:
    return sorted(
        (
            path
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file()
            and path.name not in EXCLUDED_FILES
            and not any(
                part.endswith(".egg-info")
                for part in path.relative_to(PACKAGE_ROOT).parts
            )
            and not EXCLUDED_PARTS.intersection(
                path.relative_to(PACKAGE_ROOT).parts
            )
        ),
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix(),
    )


def build_manifest() -> dict:
    files = {}
    for path in _included_files():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": "hgf_artifact_manifest",
        "file_count": len(files),
        "files": files,
    }


def write_manifest() -> None:
    MANIFEST_PATH.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_manifest() -> list[str]:
    if not MANIFEST_PATH.exists():
        return ["artifact_manifest.json is missing"]
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = build_manifest()
    errors: list[str] = []
    expected_files = expected.get("files", {})
    actual_files = actual["files"]
    for relative in sorted(set(expected_files) - set(actual_files)):
        errors.append(f"missing file: {relative}")
    for relative in sorted(set(actual_files) - set(expected_files)):
        errors.append(f"untracked file: {relative}")
    for relative in sorted(set(expected_files) & set(actual_files)):
        if expected_files[relative] != actual_files[relative]:
            errors.append(f"hash or size mismatch: {relative}")
    if expected.get("file_count") != len(expected_files):
        errors.append("manifest file_count does not match its file table")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write or verify the reproduction artifact manifest."
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        print(f"wrote {MANIFEST_PATH}")
        return
    errors = verify_manifest()
    if errors:
        raise SystemExit("\n".join(errors))
    print("artifact manifest: PASS")


if __name__ == "__main__":
    main()
