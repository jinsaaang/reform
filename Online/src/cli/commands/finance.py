"""Finance seed audits plus offline and live pipeline commands."""

from pathlib import Path
from typing import NoReturn, assert_never

import typer

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.cli.finance_input import (
    FinanceCliInputError,
    build_target as _target,
    parse_cutoff as _parse_cutoff,
    parse_mode as _parse_mode,
)
from src.cli.finance_manifest import FinanceManifestError, audit_seed
from src.cli.finance_manifest_models import SeedAuditSummary
from src.cli.finance_offline import (
    OfflineFixtureForecastProvider,
    OfflineFixtureSearchProvider,
)
from src.core.finance_pipeline import FinancePipelineAgents, FinanceReasoningPipeline
from src.core.finance_seed_repository import FinanceSeedRepository
from src.domain.finance.artifact import ForecastArm
from src.domain.finance import pipeline
from src.domain.finance.provider import FinanceRunMode, SearchSourcePolicy
from src.domain.finance.retrieval import EligibilityPolicy, InvalidTopKError, TopK
from src.integrations.finance_live import (
    DEFAULT_FORECAST_MODEL,
    DEFAULT_REASONING_EFFORT,
    LiveForecastProvider,
    LiveSearchProvider,
)
from src.services.finance_reasoning_artifact import (
    FinanceReasoningArtifactError,
    save_finance_reasoning_artifact,
)
from src.services.historical_dag_retriever import HistoricalDagRetriever

app = typer.Typer(help="Read-only finance seed audits and offline pipeline smoke tests")


def _emit_json(model: SeedAuditSummary | pipeline.PipelineResult) -> None:
    """Write stable Pydantic JSON without human logging or color noise."""
    typer.echo(model.model_dump_json())


def _fail(
    error: FinanceManifestError | FinanceCliInputError | FinanceReasoningArtifactError,
) -> NoReturn:
    """Render one actionable diagnostic on stderr and return a nonzero exit."""
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command("seed-audit")
def seed_audit(
    db: Path = typer.Option(..., "--db", help="Explicit immutable public DB path"),
    manifest: Path = typer.Option(
        ..., "--manifest", help="Explicit tracked finance seed manifest"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
) -> None:
    """Verify the pinned DB and manifest through the read-only repository."""
    try:
        summary = audit_seed(db, manifest)
    except FinanceManifestError as error:
        _fail(error)
    if json_output:
        _emit_json(summary)
    else:
        typer.echo(
            f"OK: {summary.canonical_selection.total_rows} finance episodes; "
            f"sha256={summary.asset.sha256}"
        )


@app.command("pipeline-smoke")
def pipeline_smoke(
    db: Path = typer.Option(..., "--db", help="Explicit immutable public DB path"),
    question: str = typer.Option(..., "--question", help="Current question text"),
    cutoff: str = typer.Option(..., "--cutoff", help="Aware ISO-8601 cutoff"),
    context: str = typer.Option(..., "--context", help="Current question context"),
    mode: str = typer.Option("current", "--mode", help="current or historical"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
) -> None:
    """Run the real Searcher and Forecaster with deterministic offline providers."""
    try:
        run_mode = _parse_mode(mode)
        target = _target(question, context, _parse_cutoff(cutoff))
    except FinanceCliInputError as error:
        _fail(error)
    search_provider = OfflineFixtureSearchProvider()
    forecast_provider = OfflineFixtureForecastProvider()
    repository = FinanceSeedRepository(db)
    reasoning_pipeline = FinanceReasoningPipeline(
        agents=FinancePipelineAgents(
            searcher=FinanceSearchAgent(search_provider),
            forecaster=FinanceForecastAgent(forecast_provider),
        ),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
    )
    result = reasoning_pipeline.run(
        pipeline.PipelineRunRequest(
            target_profile=target,
            run_mode=run_mode,
            source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
            eligibility_policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=TopK(3),
        )
    )
    if json_output:
        _emit_json(result)
    else:
        typer.echo(f"{result.status}: mode={run_mode.value}")
    match result:
        case pipeline.PipelineSucceeded():
            return
        case pipeline.PipelineAbstained():
            raise typer.Exit(1)
        case _ as unreachable:
            assert_never(unreachable)


@app.command("pipeline-run")
def pipeline_run(
    db: Path = typer.Option(..., "--db", help="Explicit immutable public DB path"),
    manifest: Path = typer.Option(
        ..., "--manifest", help="Explicit tracked finance seed manifest"
    ),
    question: str = typer.Option(..., "--question", help="Current question text"),
    cutoff: str = typer.Option(..., "--cutoff", help="Aware ISO-8601 cutoff"),
    context: str = typer.Option(..., "--context", help="Current question context"),
    mode: str = typer.Option("current", "--mode", help="Only current is supported"),
    top_k: int = typer.Option(3, "--top-k", min=1, help="Historical DAG bound"),
    search_result_limit: int = typer.Option(
        5,
        "--search-result-limit",
        "--result-limit",
        min=1,
        help="Maximum dated web results fetched per Searcher pass",
    ),
    artifact_dir: Path = typer.Option(
        Path("artifacts/finance/runs"),
        "--artifact-dir",
        help="Directory for complete versioned reasoning artifacts",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
) -> None:
    """Run one current-only live search/forecast experiment."""
    try:
        run_mode = _parse_mode(mode)
        if run_mode is not FinanceRunMode.CURRENT_UNRESOLVED:
            raise FinanceCliInputError(
                "live pipeline-run supports current mode only; "
                "historical live runs require frozen snapshots/archive guarantee"
            )
        target = _target(
            question,
            context,
            _parse_cutoff(cutoff),
            identifier_prefix="live-current",
        )
        if search_result_limit <= 0:
            raise FinanceCliInputError("search result limit must be positive")
        top_k_value = TopK.parse(top_k)
        _ = audit_seed(db, manifest)
    except (FinanceManifestError, FinanceCliInputError) as error:
        _fail(error)
    except (InvalidTopKError, ValueError) as error:
        _fail(FinanceCliInputError(str(error)))

    repository = FinanceSeedRepository(db)
    reasoning_pipeline = FinanceReasoningPipeline(
        agents=FinancePipelineAgents(
            searcher=FinanceSearchAgent(
                LiveSearchProvider(result_limit=search_result_limit)
            ),
            forecaster=FinanceForecastAgent(LiveForecastProvider()),
        ),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
    )
    result = reasoning_pipeline.run(
        pipeline.PipelineRunRequest(
            target_profile=target,
            run_mode=run_mode,
            source_policy=SearchSourcePolicy.LIVE_SEARCH,
            eligibility_policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=top_k_value,
        )
    )
    try:
        artifact_path = save_finance_reasoning_artifact(
            artifact_dir,
            result,
            arm=ForecastArm.SEARCH_DAG,
            forecast_model=DEFAULT_FORECAST_MODEL,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
    except FinanceReasoningArtifactError as error:
        _fail(error)
    if json_output:
        _emit_json(result)
    else:
        typer.echo(f"{result.status}: mode={run_mode.value}")
        typer.echo(f"artifact={artifact_path}")
    match result:
        case pipeline.PipelineSucceeded():
            return
        case pipeline.PipelineAbstained():
            raise typer.Exit(1)
        case _ as unreachable:
            assert_never(unreachable)


__all__ = ["app"]
