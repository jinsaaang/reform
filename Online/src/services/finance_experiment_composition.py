"""Live provider composition for the finance experiment service."""

from dataclasses import dataclass
from pathlib import Path

from src.agents.finance_reasoning_judge import JudgeProvider
from src.agents.finance_search_agent import SearchProvider
from src.domain.finance.experiment_manifest import (
    FinanceExperimentManifest,
    FinanceSearchBackend,
    SafeCompletionSettings,
    SearchSettings,
)
from src.domain.finance.provider import SearchSourcePolicy
from src.integrations.finance_completion import ProviderSeedSupport
from src.integrations.finance_live_forecast import LiveExperimentForecastProvider
from src.integrations.finance_live_judge import LiveJudgeProvider
from src.integrations.finance_live_search import LiveSearchProvider
from src.integrations.finance_public_db_search import PublicDbMetadataSearchProvider
from src.integrations.finance_rss_search import NewsRssSearch
from src.services.finance_experiment_contracts import (
    ExperimentForecastProvider,
    FinanceExperimentProviderBuildError,
    FinanceExperimentProviders,
)


@dataclass(frozen=True, slots=True)
class FinanceExperimentCompositionError(FinanceExperimentProviderBuildError):
    """The manifest cannot be represented by the live adapters."""

    def __str__(self) -> str:
        return f"finance experiment composition failed: {self.detail}"


def _same_settings_except_seed(
    expected: SafeCompletionSettings,
    actual: SafeCompletionSettings,
) -> bool:
    return (
        actual.model_copy(update={"requested_seed": expected.requested_seed})
        == expected
    )


@dataclass(frozen=True, slots=True)
class _LiveForecastProviderFactory:
    settings: SafeCompletionSettings

    def __call__(
        self,
        settings: SafeCompletionSettings,
    ) -> ExperimentForecastProvider:
        if not _same_settings_except_seed(self.settings, settings):
            raise FinanceExperimentCompositionError("forecast settings drift")
        return LiveExperimentForecastProvider(
            settings=settings,
            seed_support=ProviderSeedSupport.UNSUPPORTED,
        )


@dataclass(frozen=True, slots=True)
class _LiveJudgeProviderFactory:
    settings: tuple[SafeCompletionSettings, ...]

    def __call__(self, settings: SafeCompletionSettings) -> JudgeProvider:
        if not any(
            _same_settings_except_seed(expected, settings) for expected in self.settings
        ):
            raise FinanceExperimentCompositionError("judge settings drift")
        return LiveJudgeProvider(
            settings=settings,
            seed_support=ProviderSeedSupport.UNSUPPORTED,
        )


def _new_bing_search_provider(settings: SearchSettings) -> SearchProvider:
    timeout = settings.body_fetch_timeout_seconds
    if not timeout.is_integer():
        raise FinanceExperimentCompositionError("body timeout must be whole seconds")
    return LiveSearchProvider(
        rss_search=NewsRssSearch(),
        result_limit=settings.result_limit,
        fetch_timeout=int(timeout),
        backend=FinanceSearchBackend.BING_NEWS_RSS_V1,
    )


def _new_forecast_provider_factory(
    settings: SafeCompletionSettings,
) -> _LiveForecastProviderFactory:
    return _LiveForecastProviderFactory(settings)


def _new_judge_provider_factory(
    settings: tuple[SafeCompletionSettings, ...],
) -> _LiveJudgeProviderFactory:
    return _LiveJudgeProviderFactory(settings)


def build_live_finance_experiment_providers(
    manifest: FinanceExperimentManifest,
) -> FinanceExperimentProviders:
    """Build explicit Bing RSS and no-tool completion provider factories."""
    validated = FinanceExperimentManifest.model_validate_json(
        manifest.model_dump_json()
    )
    if (
        validated.search.source_policy is not SearchSourcePolicy.LIVE_SEARCH
        or validated.search.backend is not FinanceSearchBackend.BING_NEWS_RSS_V1
    ):
        raise FinanceExperimentCompositionError("live Bing manifest required")
    return FinanceExperimentProviders(
        search_provider=_new_bing_search_provider(validated.search),
        forecast_provider_factory=_new_forecast_provider_factory(validated.forecast),
        judge_provider_factory=_new_judge_provider_factory(
            tuple(member.settings for member in validated.judges)
        ),
    )


def build_finance_experiment_providers(
    manifest: FinanceExperimentManifest,
    db_path: Path,
) -> FinanceExperimentProviders:
    """Compose live or retrospective search from the manifest's closed policy."""
    validated = FinanceExperimentManifest.model_validate_json(
        manifest.model_dump_json()
    )
    if validated.search.source_policy is SearchSourcePolicy.LIVE_SEARCH:
        return build_live_finance_experiment_providers(validated)
    if (
        validated.search.source_policy is SearchSourcePolicy.PUBLIC_DB_METADATA
        and validated.search.backend is FinanceSearchBackend.PUBLIC_DB_METADATA_V1
    ):
        return FinanceExperimentProviders(
            search_provider=PublicDbMetadataSearchProvider(
                db_path=db_path,
                result_limit=validated.search.result_limit,
            ),
            forecast_provider_factory=_new_forecast_provider_factory(
                validated.forecast
            ),
            judge_provider_factory=_new_judge_provider_factory(
                tuple(member.settings for member in validated.judges)
            ),
        )
    raise FinanceExperimentCompositionError("unsupported experiment search policy")


__all__ = [
    "FinanceExperimentCompositionError",
    "FinanceExperimentProviders",
    "build_finance_experiment_providers",
    "build_live_finance_experiment_providers",
]
