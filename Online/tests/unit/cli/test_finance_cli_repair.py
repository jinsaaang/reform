"""Repair regressions for finance manifest integrity and CLI side effects."""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, cast

import pytest
from typer.testing import CliRunner

from src.cli.main import app
from src.utils.logging import is_finance_cli_invocation

_ROOT = Path(__file__).resolve().parents[3]
_DB = _ROOT / "data/releases/worldreasoner/v1.0.0/worldreasoner_public.db"
_MANIFEST = _ROOT / "docs/research/finance_seed_v1_manifest.json"
_MANIFEST_SHA256 = "8781718ed4bad08cabaecf820fe5b8ffe9b7d19986ad990b4f0fab359feda269"
_MANIFEST_SIZE = 30743
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow one parsed JSON value to a mapping for a fixture mutation."""
    if not isinstance(value, dict):
        raise AssertionError("manifest fixture shape is not an object")
    return value


def _sequence(value: JsonValue) -> list[JsonValue]:
    """Narrow one parsed JSON value to a list for a fixture mutation."""
    if not isinstance(value, list):
        raise AssertionError("manifest fixture shape is not a list")
    return value


def _integer(value: JsonValue) -> int:
    """Narrow one parsed JSON value to an integer for a fixture mutation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError("manifest fixture shape is not an integer")
    return value


def _manifest_copy(tmp_path: Path) -> tuple[Path, dict[str, JsonValue]]:
    """Copy the tracked manifest and parse it for one isolated mutation."""
    payload = _mapping(
        cast(JsonValue, json.loads(_MANIFEST.read_text(encoding="utf-8")))
    )
    target = tmp_path / "manifest.json"
    _ = target.write_text(json.dumps(payload), encoding="utf-8")
    return target, payload


def _audit(runner: CliRunner, manifest: Path):
    """Invoke seed-audit against the immutable DB and explicit manifest path."""
    return runner.invoke(
        app,
        [
            "finance",
            "seed-audit",
            "--db",
            str(_DB),
            "--manifest",
            str(manifest),
            "--json",
        ],
    )


def _mutate_question_id_blob(payload: dict[str, JsonValue]) -> None:
    inventory = _mapping(payload["schema_inventory"])
    tables = _sequence(inventory["tables"])
    question = _mapping(
        next(table for table in tables if _mapping(table)["name"] == "questions")
    )
    columns = _sequence(question["columns"])
    identifier = _mapping(
        next(column for column in columns if _mapping(column)["name"] == "id")
    )
    identifier["type"] = "BLOB"


def _mutate_not_null(payload: dict[str, JsonValue]) -> None:
    inventory = _mapping(payload["schema_inventory"])
    tables = _sequence(inventory["tables"])
    question = _mapping(
        next(table for table in tables if _mapping(table)["name"] == "questions")
    )
    columns = _sequence(question["columns"])
    identifier = _mapping(
        next(column for column in columns if _mapping(column)["name"] == "id")
    )
    identifier["not_null"] = True


def _mutate_remove_articles_column(payload: dict[str, JsonValue]) -> None:
    inventory = _mapping(payload["schema_inventory"])
    tables = _sequence(inventory["tables"])
    articles = _mapping(
        next(table for table in tables if _mapping(table)["name"] == "articles")
    )
    _ = _sequence(articles["columns"]).pop()


def _mutate_index_sql(payload: dict[str, JsonValue]) -> None:
    inventory = _mapping(payload["schema_inventory"])
    indexes = _sequence(inventory["indexes_views_triggers"])
    _mapping(indexes[0])["sql"] = "tampered-index-sql"


def _mutate_forecast_rows(payload: dict[str, JsonValue]) -> None:
    linkage = _mapping(payload["graph_linkage"])
    forecasts = _mapping(linkage["forecasts"])
    forecasts["rows"] = _integer(forecasts["rows"]) + 1


@pytest.mark.parametrize(
    ("name", "mutator"),
    [
        ("question-id-type", _mutate_question_id_blob),
        ("question-not-null", _mutate_not_null),
        ("articles-column", _mutate_remove_articles_column),
        ("index-sql", _mutate_index_sql),
        ("forecast-rows", _mutate_forecast_rows),
    ],
)
def test_seed_audit_rejects_each_manifest_metadata_tamper(
    tmp_path: Path,
    name: str,
    mutator: Callable[[dict[str, JsonValue]], None],
) -> None:
    """Given one changed manifest metadata field, audit exits nonzero."""
    manifest, payload = _manifest_copy(tmp_path)
    mutator(payload)
    _ = manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = _audit(CliRunner(), manifest)

    assert result.exit_code == 1, name


def test_seed_audit_accepts_byte_identical_manifest_copy(tmp_path: Path) -> None:
    """Given a byte-identical manifest at another path, audit succeeds."""
    target = tmp_path / "manifest-copy.json"
    _ = target.write_bytes(_MANIFEST.read_bytes())

    result = _audit(CliRunner(), target)

    assert result.exit_code == 0
    payload = cast(JsonValue, json.loads(result.stdout))
    payload_mapping = _mapping(payload)
    assert payload_mapping["manifest_sha256"] == _MANIFEST_SHA256
    assert payload_mapping["manifest_size_bytes"] == _MANIFEST_SIZE


def test_seed_audit_rejects_malformed_manifest_json(tmp_path: Path) -> None:
    """Given malformed manifest bytes, audit fails closed."""
    target = tmp_path / "malformed.json"
    _ = target.write_text("{not-json", encoding="utf-8")

    result = _audit(CliRunner(), target)

    assert result.exit_code == 1


def _run_real_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the installed entry point with provider keys scrubbed."""
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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    executable = shutil.which("wr")
    if executable is None:
        raise AssertionError("wr entry point is unavailable in the test environment")
    return subprocess.run(
        [executable, *args],
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_finance_real_cli_has_no_logs_and_stable_full_output(tmp_path: Path) -> None:
    """Given repeated real seed-audit runs, no logs are added and bytes match."""
    before = set((_ROOT / "logs").glob("worldreasoner_*.log"))
    args = (
        "finance",
        "seed-audit",
        "--db",
        str(_DB),
        "--manifest",
        str(_MANIFEST),
        "--json",
    )
    try:
        first = _run_real_cli(*args, cwd=tmp_path)
        middle = set((_ROOT / "logs").glob("worldreasoner_*.log"))
        second = _run_real_cli(*args, cwd=tmp_path)
        after = set((_ROOT / "logs").glob("worldreasoner_*.log"))
        assert first.returncode == second.returncode == 0
        assert first.stdout + first.stderr == second.stdout + second.stderr
        assert after == before
        assert middle == before
        assert not (tmp_path / "worldreasoner.db").exists()
        assert not (tmp_path / "worldreasoner.db-wal").exists()
        assert not (tmp_path / "worldreasoner.db-shm").exists()
    finally:
        for created in set((_ROOT / "logs").glob("worldreasoner_*.log")) - before:
            created.unlink(missing_ok=True)


def test_root_and_non_finance_help_characterization() -> None:
    """Given current registration, root and upstream non-finance help remain present."""
    runner = CliRunner()

    root = runner.invoke(app, ["--help"])
    db = runner.invoke(app, ["db", "--help"])

    assert root.exit_code == db.exit_code == 0
    assert all(command in root.stdout for command in ("db", "question", "finance"))
    assert all(command in db.stdout for command in ("stats", "list", "show"))


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("wr", "finance", "seed-audit"), True),
        (("python", "-m", "src.cli.main", "finance", "--help"), False),
        (("pytest", "tests/unit/cli/test_finance_cli.py"), False),
        (("wr", "db", "stats"), False),
        (("wr", "--verbose", "finance", "pipeline-smoke"), True),
    ],
)
def test_finance_argv_detection_is_narrow(
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    """Given common entrypoint argv shapes, only root finance is suppressed."""
    assert is_finance_cli_invocation(argv) is expected


def test_manifest_digest_constant_matches_tracked_bytes() -> None:
    """Given the tracked manifest, its pinned digest and size are exact."""
    data = _MANIFEST.read_bytes()

    assert len(data) == _MANIFEST_SIZE
    assert hashlib.sha256(data).hexdigest() == _MANIFEST_SHA256
