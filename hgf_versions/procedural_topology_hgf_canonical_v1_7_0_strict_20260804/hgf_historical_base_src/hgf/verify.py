"""Verify the public input bundle and runtime architecture."""

from __future__ import annotations

import ast
from pathlib import Path

from .manifest import verify_manifest
from .package import PACKAGE_ROOT
from .preflight import run_preflight


def source_dependency_violations(path: Path) -> list[str]:
    """Return external-runner imports and import-path mutations in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    external_package = "".join(("fore", "caster"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == external_package:
                    violations.append(
                        f"{path.name}:{node.lineno} imports {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] == external_package:
                violations.append(
                    f"{path.name}:{node.lineno} imports from {module!r}"
                )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "path"
        ):
            violations.append(f"{path.name}:{node.lineno} accesses sys.path")

    return violations


def run_checks() -> list[str]:
    errors = verify_manifest()
    report = run_preflight(validate_evidence=False)
    errors.extend(report["errors"])

    source_root = PACKAGE_ROOT / "src" / "hgf"
    legacy_dotted = ".".join(("legacy", "original_hgf"))
    legacy_slash = "/".join(("legacy", "original_hgf"))
    for path in source_root.glob("*.py"):
        errors.extend(source_dependency_violations(path))
        source = path.read_text(encoding="utf-8")
        if legacy_dotted in source or legacy_slash in source:
            errors.append(f"{path.name} imports or opens legacy HGF state")

    runner = (source_root / "baselines.py").read_text(encoding="utf-8")
    for expression in (
        'fixed_case["retrieved_memory_question_id"]',
        'fixed_case["worked_exemplar"]',
        "load_hgf_blueprint_bank",
        "load_factor_blueprint_bank",
        '"probability_postprocessing": "none"',
    ):
        if expression not in runner:
            errors.append(f"runner is missing fixed input: {expression}")
    return errors


def main() -> None:
    errors = run_checks()
    if errors:
        raise SystemExit("\n".join(f"FAIL: {error}" for error in errors))
    print("HGF public bundle: PASS")
    print("200 HGF memories; 100 fixed cases; seven method inputs verified")


if __name__ == "__main__":
    main()
