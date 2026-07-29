from __future__ import annotations

import argparse
import hashlib
import json
import os
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
EXCLUDED_FILE_SUFFIXES = (
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
)
BINARY_SUFFIXES = {".sqlite"}


def _included_files() -> list[Path]:
    included: list[Path] = []
    for root, directories, files in os.walk(PACKAGE_ROOT):
        directories[:] = [
            name
            for name in directories
            if name not in EXCLUDED_PARTS
            and not name.endswith(".egg-info")
        ]
        root_path = Path(root)
        included.extend(
            root_path / name
            for name in files
            if name not in EXCLUDED_FILES
            and not name.endswith(EXCLUDED_FILE_SUFFIXES)
        )
    return sorted(
        included,
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix(),
    )


def _file_record(path: Path) -> dict[str, int | str]:
    content = path.read_bytes()
    if path.suffix not in BINARY_SUFFIXES:
        content = content.replace(b"\r\n", b"\n")
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_manifest() -> dict:
    files = {}
    for path in _included_files():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        files[relative] = _file_record(path)
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
