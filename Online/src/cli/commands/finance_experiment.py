"""Thin CLI adapters for persisted finance experiments and analysis."""

import json
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping, NoReturn
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict

from src.cli.finance_manifest import FinanceManifestError, audit_seed
from src.core.finance_seed_repository import FinanceSeedRepository
from src.services.finance_backtest import (
    FinanceBacktestError,
    analyze_finance_backtest,
    build_finance_backtest_manifest,
    write_finance_backtest_manifest,
)
from src.services.finance_experiment import (
    FinanceExperimentDependencies,
    FinanceExperimentExitCode,
    FinanceExperimentRunRequest,
    FinanceExperimentRunResult,
    run_finance_experiment,
)
from src.services.finance_experiment_composition import (
    build_finance_experiment_providers,
)
from src.services.finance_experiment_manifest import (
    FinanceManifestReadError,
    FinanceManifestValidationError,
    load_finance_experiment_manifest,
)
from src.services.finance_hashed_bundle import HashedBundleReceipt
from src.services.finance_resolution_analysis import (
    FinanceResolutionAnalysisError,
    analyze_finance_resolution,
)
from src.services.finance_seed_identity import FinanceSeedIdentityVerifier
from src.services.historical_dag_retriever import HistoricalDagRetriever


def _dependencies(db: Path) -> FinanceExperimentDependencies:
    """Assemble manifest-selected dependencies after DB preflight."""
    repository = FinanceSeedRepository(db)
    return FinanceExperimentDependencies(
        provider_builder=lambda manifest: build_finance_experiment_providers(
            manifest,
            db,
        ),
        identity_verifier=FinanceSeedIdentityVerifier(db),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
        clock=lambda: datetime.now(UTC),
        monotonic_clock=time.monotonic,
        uuid_factory=uuid4,
        alias_salt_factory=lambda: secrets.token_bytes(32),
    )


class _RunReceipt(BaseModel):
    """Stable receipt fields rendered by the run adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: str
    data_sha256: str
    report_sha256: str


class _RunOutput(BaseModel):
    """Machine-readable run status without provider payloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    recorded: bool
    exit_code: int
    suite_id: str | None
    trial_count: int
    receipt: _RunReceipt | None
    failure_class: str | None


class _AnalysisOutput(BaseModel):
    """Machine-readable receipt for a distinct post-resolution bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    recorded: bool
    directory: str
    data_sha256: str
    report_sha256: str


_EXIT_CODES: Final[Mapping[FinanceExperimentExitCode, int]] = MappingProxyType(
    {
        FinanceExperimentExitCode.SUCCESS: 0,
        FinanceExperimentExitCode.PARTIAL_RECORDED: 1,
        FinanceExperimentExitCode.FAILED_RECORDED: 2,
        FinanceExperimentExitCode.PREFLIGHT_FAILED: 1,
        FinanceExperimentExitCode.PERSISTENCE_FAILED: 1,
    }
)


def _fail(error: Exception) -> NoReturn:
    """Render one actionable CLI diagnostic and return exit status one."""
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _receipt_model(receipt: HashedBundleReceipt) -> _RunReceipt:
    return _RunReceipt(
        directory=str(receipt.directory),
        data_sha256=receipt.data_sha256,
        report_sha256=receipt.report_sha256,
    )


def _run_output(result: FinanceExperimentRunResult, trial_count: int) -> _RunOutput:
    """Build stable JSON and human-output data from one service result."""
    status = (
        "preflight_failed" if result.suite_status is None else result.suite_status.value
    )
    return _RunOutput(
        status=status,
        recorded=result.recorded,
        exit_code=int(result.exit_code),
        suite_id=None if result.suite_id is None else str(result.suite_id),
        trial_count=trial_count,
        receipt=None if result.receipt is None else _receipt_model(result.receipt),
        failure_class=result.failure_class,
    )


def _render_run(output: _RunOutput, json_output: bool) -> None:
    if json_output:
        typer.echo(output.model_dump_json())
        return
    typer.echo(
        f"{output.status}: recorded={str(output.recorded).lower()} "
        f"trials={output.trial_count}"
    )
    if output.receipt is not None:
        typer.echo(f"bundle={output.receipt.directory}")
        typer.echo(f"suite_sha256={output.receipt.data_sha256}")
        typer.echo(f"report_sha256={output.receipt.report_sha256}")
    if output.failure_class is not None:
        typer.echo(f"failure={output.failure_class}")


def _render_analysis(receipt: HashedBundleReceipt, json_output: bool) -> None:
    output = _AnalysisOutput(
        status="complete",
        recorded=True,
        directory=str(receipt.directory),
        data_sha256=receipt.data_sha256,
        report_sha256=receipt.report_sha256,
    )
    if json_output:
        typer.echo(output.model_dump_json())
        return
    typer.echo("complete: recorded=true")
    typer.echo(f"bundle={output.directory}")
    typer.echo(f"analysis_sha256={output.data_sha256}")
    typer.echo(f"report_sha256={output.report_sha256}")


def _exit_code(result: FinanceExperimentRunResult) -> int:
    return _EXIT_CODES[result.exit_code]


def _question_ids(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise FinanceBacktestError("unable to read exclusion file") from error
    identifiers = frozenset(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not identifiers:
        raise FinanceBacktestError("exclusion file contains no question IDs")
    return identifiers


def register_finance_experiment_commands(finance_app: typer.Typer) -> None:
    """Register experiment commands on the existing finance command group."""

    def experiment_run(
        db: Path = typer.Option(..., "--db", help="Explicit immutable public DB path"),
        seed_manifest: Path = typer.Option(
            ...,
            "--seed-manifest",
            help="Pinned immutable DB seed manifest",
        ),
        experiment_manifest: Path = typer.Option(
            ...,
            "--experiment-manifest",
            "--manifest",
            help="Versioned finance experiment manifest",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            "--destination",
            help="New directory for suite.json/report.md/SHA256SUMS",
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Emit machine-readable JSON"
        ),
    ) -> None:
        """Run the manifest-bound Direct/Search-only/Search+DAG experiment."""
        try:
            _ = audit_seed(db, seed_manifest)
            manifest = load_finance_experiment_manifest(experiment_manifest)
        except (
            FinanceManifestError,
            FinanceManifestReadError,
            FinanceManifestValidationError,
        ) as error:
            _fail(error)
        result = run_finance_experiment(
            FinanceExperimentRunRequest(manifest, output_dir),
            _dependencies(db),
        )
        output = _run_output(
            result,
            len(manifest.questions) * manifest.repetitions,
        )
        _render_run(output, json_output)
        raise typer.Exit(_exit_code(result))

    def experiment_analyze(
        suite: Path = typer.Option(
            ...,
            "--suite",
            "--source-bundle",
            help="Verified ex-ante experiment bundle directory",
        ),
        resolution_manifest: Path = typer.Option(
            ..., "--resolution-manifest", help="Outcome-only resolution manifest"
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            "--destination",
            help="New directory for analysis.json/report.md/SHA256SUMS",
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Emit machine-readable JSON"
        ),
    ) -> None:
        """Analyze outcomes separately from the verified ex-ante suite."""
        try:
            receipt = analyze_finance_resolution(
                suite,
                resolution_manifest,
                output_dir,
            )
        except (FinanceResolutionAnalysisError, OSError) as error:
            _fail(error)
        _render_analysis(receipt, json_output)

    def backtest_prepare(
        db: Path = typer.Option(..., "--db", help="Immutable public DB path"),
        seed_manifest: Path = typer.Option(
            ...,
            "--seed-manifest",
            help="Pinned immutable DB seed manifest",
        ),
        template_manifest: Path = typer.Option(
            ...,
            "--template-manifest",
            help="Experiment settings template; its questions are replaced",
        ),
        output_file: Path = typer.Option(
            ...,
            "--output-file",
            help="New ground-truth-free historical manifest path",
        ),
        manifest_id: str = typer.Option(
            "finance-resolved-backtest-v1",
            "--manifest-id",
            help="Unique experiment manifest identifier",
        ),
        limit: int | None = typer.Option(
            None,
            "--limit",
            min=1,
            help="Optional deterministic target limit",
        ),
        model: str = typer.Option(
            "openrouter/openai/o4-mini",
            "--model",
            help="Reasoning model used by both forecaster and blind judges",
        ),
        exclude_question_file: Path | None = typer.Option(
            None,
            "--exclude-question-file",
            help="Optional newline-delimited question IDs to exclude",
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Emit machine-readable JSON"
        ),
    ) -> None:
        """Prepare resolved binary targets without exposing their outcomes."""
        try:
            _ = audit_seed(db, seed_manifest)
            template = load_finance_experiment_manifest(template_manifest)
            manifest = build_finance_backtest_manifest(
                db,
                template,
                manifest_id,
                limit,
                model,
                _question_ids(exclude_question_file),
            )
            write_finance_backtest_manifest(output_file, manifest)
        except (
            FinanceBacktestError,
            FinanceManifestError,
            FinanceManifestReadError,
            FinanceManifestValidationError,
        ) as error:
            _fail(error)
        output = {
            "status": "prepared",
            "manifest": str(output_file),
            "question_count": len(manifest.questions),
            "model": manifest.forecast.model,
            "outcomes_exposed": False,
        }
        if json_output:
            typer.echo(json.dumps(output, separators=(",", ":")))
            return
        typer.echo(
            f"prepared: questions={len(manifest.questions)} outcomes_exposed=false"
        )
        typer.echo(f"manifest={output_file}")

    def backtest_analyze(
        suite: Path = typer.Option(
            ...,
            "--suite",
            "--source-bundle",
            help="Completed historical experiment bundle",
        ),
        db: Path = typer.Option(..., "--db", help="Immutable public DB path"),
        seed_manifest: Path = typer.Option(
            ...,
            "--seed-manifest",
            help="Pinned immutable DB seed manifest",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            "--destination",
            help="New resolved analysis bundle directory",
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Emit machine-readable JSON"
        ),
    ) -> None:
        """Join pinned outcomes after the run and score the historical suite."""
        try:
            _ = audit_seed(db, seed_manifest)
            receipt = analyze_finance_backtest(suite, db, output_dir)
        except (
            FinanceBacktestError,
            FinanceManifestError,
            FinanceResolutionAnalysisError,
            OSError,
        ) as error:
            _fail(error)
        _render_analysis(receipt, json_output)

    finance_app.command("experiment-run")(experiment_run)
    finance_app.command("experiment-analyze")(experiment_analyze)
    finance_app.command("backtest-prepare")(backtest_prepare)
    finance_app.command("backtest-analyze")(backtest_analyze)


__all__ = ["register_finance_experiment_commands"]
