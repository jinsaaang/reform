#!/usr/bin/env python3
"""Report imported packages and forbidden path or result dependencies."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [ROOT / "hgf_final", ROOT / "scripts"]
INTERNAL = {
    "hgf",
    "hgf_e2e_topology",
    "hgf_original_input_adapter",
    "hgf_e2e_topology_provider_pinned",
    "hgf_e2e_topology_sidecar",
}
THIRD_PARTY = {"dotenv", "openai", "pydantic"}
FORBIDDEN = re.compile(
    "|".join(("/" + "home/", "/" + "tmp/", "FINAL_RESULTS" + ".json", "runs/.+results"))
)


def main() -> int:
    imports: set[str] = set()
    violations: list[dict[str, object]] = []
    files = sorted(
        path
        for root in SCAN_ROOTS
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".", 1)[0])
        inspected_lines = [] if path.resolve() == Path(__file__).resolve() else source.splitlines()
        for line_number, line in enumerate(inspected_lines, start=1):
            if FORBIDDEN.search(line):
                violations.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "text": line.strip(),
                    }
                )
    standard = set(getattr(sys, "stdlib_module_names", set())) | {"__future__"}
    unknown = sorted(imports - standard - INTERNAL - THIRD_PARTY)
    report = {
        "status": "passed" if not unknown and not violations else "failed",
        "python_files": len(files),
        "internal_packages": sorted(imports & INTERNAL),
        "third_party_packages": sorted(imports & THIRD_PARTY),
        "unknown_packages": unknown,
        "forbidden_dependencies": violations,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
