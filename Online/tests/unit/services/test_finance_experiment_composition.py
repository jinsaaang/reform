"""Live finance experiment composition is manifest-pinned."""

from pathlib import Path

import pytest

from src.domain.finance.experiment_manifest import (
    FinanceSearchBackend,
    SafeCompletionSettings,
    SearchSettings,
)
from src.services import finance_experiment_composition as composition
from src.services.finance_experiment_manifest import (
    load_finance_experiment_manifest,
)
from tests.fixtures.finance_experiment_service import (
    OfflineExperimentState,
    OfflineForecastFactory,
    OfflineJudgeFactory,
    OfflineSearchProvider,
)

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs/experiments/finance_live_10_2026-07-18.json"
)


def test_public_composition_pins_bing_and_exact_completion_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_finance_experiment_manifest(_MANIFEST_PATH)
    state = OfflineExperimentState()
    search_provider = OfflineSearchProvider(state, '{"candidate_payloads":[]}')
    forecast_factory = OfflineForecastFactory(state)
    judge_factory = OfflineJudgeFactory(state)
    search_settings: list[SearchSettings] = []
    forecast_settings: list[SafeCompletionSettings] = []
    judge_settings: list[tuple[SafeCompletionSettings, ...]] = []

    def capture_search(settings: SearchSettings) -> OfflineSearchProvider:
        search_settings.append(settings)
        return search_provider

    def capture_forecast(
        settings: SafeCompletionSettings,
    ) -> OfflineForecastFactory:
        forecast_settings.append(settings)
        return forecast_factory

    def capture_judges(
        settings: tuple[SafeCompletionSettings, ...],
    ) -> OfflineJudgeFactory:
        judge_settings.append(settings)
        return judge_factory

    monkeypatch.setenv("SEARXNG_BASE_URL", "https://conflict.invalid")
    monkeypatch.setattr(composition, "_new_bing_search_provider", capture_search)
    monkeypatch.setattr(
        composition,
        "_new_forecast_provider_factory",
        capture_forecast,
    )
    monkeypatch.setattr(
        composition,
        "_new_judge_provider_factory",
        capture_judges,
    )

    providers = composition.build_live_finance_experiment_providers(manifest)

    assert providers.search_provider is search_provider
    assert providers.forecast_provider_factory is forecast_factory
    assert providers.judge_provider_factory is judge_factory
    assert search_settings == [manifest.search]
    assert search_settings[0].backend is FinanceSearchBackend.BING_NEWS_RSS_V1
    assert search_settings[0].result_limit == 5
    assert search_settings[0].body_fetch_timeout_seconds == 20
    assert forecast_settings == [manifest.forecast]
    assert judge_settings == [tuple(member.settings for member in manifest.judges)]
    assert state.search_requests == []
    assert state.forecast_inputs == []
    assert state.judge_payloads == []
