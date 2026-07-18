"""Behavioral tests for the isolated non-graph-building Forecaster."""

from datetime import datetime, timezone

from src.agents.finance_forecast_agent import (
    FinanceForecastAgent,
    ForecastAgentAbstained,
    ForecastAgentResult,
    ForecastAgentSucceeded,
    ForecastProvider,
    ForecastProviderError,
)
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.memory import EvidenceId
from src.domain.finance.pipeline import ForecastAbstentionReason
from src.domain.finance.search import EvidenceDirection, EvidenceItem, EvidencePack
from tests.fixtures.finance_experiment_runner import MemoryAwareForecastProvider
from tests.fixtures.finance_pipeline import FixtureForecastProvider, make_target
from tests.unit.domain.finance._factories import make_episode


def _forecast_input() -> ForecasterInput:
    episode = make_episode()
    evidence = EvidenceItem(
        evidence_id=EvidenceId("evidence-1"),
        claim="GPU demand remains elevated.",
        citation="fixture://filing/nvidia-quarter",
        available_at=datetime(2024, 12, 1, tzinfo=timezone.utc),
        retrieved_at=datetime(2024, 12, 2, tzinfo=timezone.utc),
        content_hash="sha256:fixture-evidence-1",
        direction=EvidenceDirection.SUPPORTS,
        context_slot="demand",
    )
    return ForecasterInput(
        target_profile=make_target(),
        evidence_pack=EvidencePack(
            items=(evidence,),
            historical_dag_references=(episode.reference,),
        ),
        historical_memory=(episode,),
    )


def _succeeded(result: ForecastAgentResult) -> ForecastAgentSucceeded:
    if isinstance(result, ForecastAgentSucceeded):
        return result
    raise AssertionError(result.reason)


class _FailingForecastProvider:
    def forecast(self, forecast_input: ForecasterInput) -> str:
        del forecast_input
        raise ForecastProviderError(code="fixture_failure", detail="offline failure")


class TestForecastProviderBoundary:
    def test_should_pass_only_validated_forecaster_input_to_provider(self) -> None:
        # Given: an admitted evidence pack and selected immutable history
        provider = FixtureForecastProvider()
        agent = FinanceForecastAgent(provider)

        # When: the non-graph-building Forecaster invokes its provider
        result = _succeeded(agent.forecast(_forecast_input()))

        # Then: provider input has exactly the authorized three-field surface
        assert result.forecast_result.outcome_probabilities[0].probability == 0.5
        provider_input = provider.inputs[0]
        assert set(type(provider_input).model_fields) == {
            "target_profile",
            "evidence_pack",
            "historical_memory",
        }
        serialized = provider_input.model_dump_json()
        assert "ground_truth" not in serialized and "current_dag" not in serialized
        assert '"historical_outcome"' in serialized

    def test_should_expose_no_search_retrieval_or_graph_provider_capability(
        self,
    ) -> None:
        # Given: the structural Forecast provider contract
        public_names = {
            name for name in ForecastProvider.__dict__ if not name.startswith("_")
        }

        # When: its callable surface is inspected
        # Then: forecast is its sole external capability
        assert public_names == {"forecast"}

    def test_should_abstain_on_malformed_provider_output_without_default(self) -> None:
        # Given: a provider response that cannot become ForecastResult
        provider = FixtureForecastProvider(serialized_output="not-json")
        agent = FinanceForecastAgent(provider)

        # When: the response crosses the Forecaster output boundary
        result = agent.forecast(_forecast_input())

        # Then: validation yields a typed abstention, not a coerced probability
        assert isinstance(result, ForecastAgentAbstained)
        assert result.reason is ForecastAbstentionReason.MALFORMED_OUTPUT

    def test_should_abstain_on_out_of_range_probability(self) -> None:
        # Given: otherwise valid output with one final probability above one
        valid = FixtureForecastProvider().forecast(_forecast_input())
        invalid = valid.replace('"probability":0.5', '"probability":1.2', 1)
        agent = FinanceForecastAgent(FixtureForecastProvider(serialized_output=invalid))

        # When: the invalid distribution crosses the provider boundary
        result = agent.forecast(_forecast_input())

        # Then: domain validation becomes an explicit abstention
        assert isinstance(result, ForecastAgentAbstained)
        assert result.reason is ForecastAbstentionReason.MALFORMED_OUTPUT

    def test_should_convert_typed_provider_error_to_explicit_abstention(self) -> None:
        # Given: a provider-specific typed failure
        agent = FinanceForecastAgent(_FailingForecastProvider())

        # When: the provider is invoked
        result = agent.forecast(_forecast_input())

        # Then: the provider error is explicit and contains no partial result
        assert isinstance(result, ForecastAgentAbstained)
        assert result.reason is ForecastAbstentionReason.PROVIDER_ERROR
        assert "fixture_failure" in result.detail

    def test_pack_metadata_does_not_authorize_missing_historical_memory(self) -> None:
        # Given: pack metadata names a DAG that is absent from actual memory
        forecast_input = _forecast_input().model_copy(update={"historical_memory": ()})
        agent = FinanceForecastAgent(FixtureForecastProvider())

        # When
        result = agent.forecast(forecast_input)

        # Then
        assert isinstance(result, ForecastAgentAbstained)
        assert result.reason is ForecastAbstentionReason.UNKNOWN_HISTORICAL_REFERENCE

    def test_should_authorize_scenario_history_from_actual_memory(self) -> None:
        # Given: fixed evidence carries no DAG references while C receives memory
        original = _forecast_input()
        fixed_pack = original.evidence_pack.model_copy(
            update={"historical_dag_references": ()}
        )
        forecast_input = original.model_copy(update={"evidence_pack": fixed_pack})
        provider = MemoryAwareForecastProvider([])
        agent = FinanceForecastAgent(provider)

        # When
        result = agent.forecast(forecast_input)

        # Then: the scenario may cite only the supplied historical memory
        succeeded = _succeeded(result)
        assert succeeded.forecast_result.scenarios[0].historical_dag_references == (
            original.historical_memory[0].reference,
        )
