"""Executable identity and logging-preservation regressions for finance CLI startup."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.logging import is_finance_cli_invocation

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("/abs/.venv/bin/wr", "finance", "--help"), True),
        (("wr.exe", "finance", "--help"), True),
        (("python", "-c", "finance"), False),
        (("pytest", "finance"), False),
        (("mytool", "finance"), False),
        (("wr", "db", "finance"), False),
        (("wr", "--verbose", "db", "finance"), False),
        (("wr", "db", "--finance"), False),
    ],
)
def test_finance_predicate_requires_wr_root_subcommand(
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    """Given arbitrary executable/argument shapes, only root finance matches."""
    assert is_finance_cli_invocation(argv) is expected


def test_python_c_finance_preserves_logging_handlers(tmp_path: Path) -> None:
    """Given python ``-c`` with a finance argument, handlers stay configured."""
    program = (
        "import sys\n"
        "from loguru import logger\n"
        "from src.utils.logging import is_finance_cli_invocation\n"
        "print(is_finance_cli_invocation(tuple(sys.argv)), "
        "len(logger._core.handlers))\n"
    )
    env = os.environ.copy()
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "PERPLEXITY_API_KEY",
        "SERPER_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        _ = env.pop(key, None)
    before = set((_ROOT / "logs").glob("worldreasoner_*.log"))
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                "finance",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0
        output = result.stdout.decode("utf-8").strip().split()
        assert output[0] == "False"
        assert int(output[1]) > 0
    finally:
        for created in set((_ROOT / "logs").glob("worldreasoner_*.log")) - before:
            created.unlink(missing_ok=True)
