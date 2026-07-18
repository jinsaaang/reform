#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
SEED_DB="data/releases/worldreasoner/v1.0.0/worldreasoner_public.db"

cd "$REPO_ROOT"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    printf 'error: uv is required; install it from https://docs.astral.sh/uv/\n' >&2
    exit 1
fi

if [[ ! -r "$SEED_DB" ]]; then
    printf 'error: missing seed DB at %s; follow docs/research/financial_worldreasoner_bootstrap.md\n' \
        "$SEED_DB" >&2
    exit 1
fi

SOURCE_PATHS=(
    src/cli/main.py
    src/config/constants.py
    src/core/llm.py
    src/tools/base/output_models.py
    src/tools/collectors/rss_fetch.py
    src/tools/collectors/web_fetch.py
    src/utils/logging.py
    src/agents/finance_forecast_agent.py
    src/agents/finance_reasoning_judge.py
    src/agents/finance_search_agent.py
    src/cli/commands/finance.py
    src/cli/commands/finance_experiment.py
    src/cli/finance_input.py
    src/cli/finance_manifest.py
    src/cli/finance_manifest_models.py
    src/cli/finance_manifest_spec.py
    src/cli/finance_offline.py
    src/core/finance_experiment_runner.py
    src/core/finance_experiment_runtime.py
    src/core/finance_judge_panel.py
    src/core/finance_pipeline.py
    src/core/finance_seed_repository.py
    src/core/llm_response.py
    src/core/structured_completion.py
    src/domain/finance
    src/integrations/finance_completion.py
    src/integrations/finance_live.py
    src/integrations/finance_live_forecast.py
    src/integrations/finance_live_judge.py
    src/integrations/finance_live_search.py
    src/integrations/finance_public_db_search.py
    src/integrations/finance_rss_search.py
    src/services/finance_aliases.py
    src/services/finance_artifact_sanitizer.py
    src/services/finance_atomic_publish.py
    src/services/finance_backtest.py
    src/services/finance_experiment.py
    src/services/finance_experiment_arm.py
    src/services/finance_experiment_artifact.py
    src/services/finance_experiment_batch.py
    src/services/finance_experiment_composition.py
    src/services/finance_experiment_contracts.py
    src/services/finance_experiment_manifest.py
    src/services/finance_experiment_panels.py
    src/services/finance_experiment_recording.py
    src/services/finance_experiment_records.py
    src/services/finance_experiment_reporting.py
    src/services/finance_experiment_trial.py
    src/services/finance_hashed_bundle.py
    src/services/finance_judge_alias_sources.py
    src/services/finance_judge_candidate_view.py
    src/services/finance_judge_memory_view.py
    src/services/finance_judge_panel.py
    src/services/finance_judge_source_validation.py
    src/services/finance_judge_view.py
    src/services/finance_reasoning_artifact.py
    src/services/finance_resolution_analysis.py
    src/services/finance_resolution_metrics.py
    src/services/finance_resolution_reporting.py
    src/services/finance_seed_identity.py
    src/services/historical_dag_retriever.py
)

TEST_PATHS=(
    tests/integration/test_finance_experiment_offline.py
    tests/integration/test_finance_resolved_backtest.py
    tests/unit/agents/test_finance_forecast_agent.py
    tests/unit/agents/test_finance_reasoning_judge.py
    tests/unit/agents/test_finance_search_agent.py
    tests/unit/agents/test_finance_search_provenance.py
    tests/unit/cli
    tests/unit/core/test_finance_experiment_runner.py
    tests/unit/core/test_finance_judge_panel.py
    tests/unit/core/test_finance_pipeline.py
    tests/unit/core/test_finance_pipeline_integrity.py
    tests/unit/core/test_finance_seed_repository.py
    tests/unit/domain/finance
    tests/unit/integrations/test_finance_live.py
    tests/unit/integrations/test_finance_live_backend.py
    tests/unit/integrations/test_finance_live_judge.py
    tests/unit/services
    tests/unit/tools/test_web_fetch.py::test_fast_fetch_extracts_readable_html_without_browser
)

printf 'finance-check: ruff lint\n'
"$UV_BIN" run --locked --with ruff ruff check --no-cache \
    "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]%%::*}"

printf 'finance-check: ruff format\n'
"$UV_BIN" run --locked --with ruff ruff format --check --no-cache \
    "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]%%::*}"

printf 'finance-check: strict type check\n'
"$UV_BIN" run --locked --with basedpyright basedpyright \
    --project configs/quality/basedpyright-finance-experiment.json

printf 'finance-check: finance tests\n'
"$UV_BIN" run --locked pytest -q "${TEST_PATHS[@]}"

printf 'finance-check: PASS\n'
