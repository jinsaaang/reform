"""Non-graph-building finance Forecaster with one provider capability."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.forecast import ForecasterInput, ForecastResult
from src.domain.finance.pipeline import ForecastAbstentionReason


class ForecastProvider(Protocol):
    """Only capability available to the finance Forecaster."""

    def forecast(self, forecast_input: ForecasterInput) -> str:
        """Return a serialized forecast boundary."""
        ...


@dataclass(frozen=True, slots=True)
class ForecastProviderError(Exception):
    """Expected typed failure raised by a Forecast provider adapter."""

    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"forecast provider {self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ForecastAgentSucceeded:
    """Validated provider forecast."""

    forecast_result: ForecastResult


@dataclass(frozen=True, slots=True)
class ForecastAgentAbstained:
    """Typed fail-closed provider or validation outcome."""

    reason: ForecastAbstentionReason
    detail: str


ForecastAgentResult: TypeAlias = ForecastAgentSucceeded | ForecastAgentAbstained


@dataclass(frozen=True, slots=True)
class FinanceForecastAgent:
    """Validate forecast output without search, retrieval, or graph capability."""

    provider: ForecastProvider

    def forecast(self, forecast_input: ForecasterInput) -> ForecastAgentResult:
        """Invoke the isolated provider with only validated ForecasterInput."""
        try:
            serialized = self.provider.forecast(forecast_input)
        except ForecastProviderError as error:
            return ForecastAgentAbstained(
                reason=ForecastAbstentionReason.PROVIDER_ERROR,
                detail=str(error),
            )
        try:
            result = ForecastResult.model_validate_json(serialized)
        except ValidationError:
            return ForecastAgentAbstained(
                reason=ForecastAbstentionReason.MALFORMED_OUTPUT,
                detail="provider output failed ForecastResult validation",
            )
        expected_labels = set(forecast_input.target_profile.outcome_space)
        actual_labels = {str(item.label) for item in result.outcome_probabilities}
        if actual_labels != expected_labels:
            return ForecastAgentAbstained(
                reason=ForecastAbstentionReason.OUTCOME_SPACE_MISMATCH,
                detail="provider outcome labels differ from target outcome space",
            )
        evidence_ids = {item.evidence_id for item in forecast_input.evidence_pack.items}
        referenced_evidence = {
            evidence_id
            for scenario in result.scenarios
            for evidence_id in scenario.evidence_ids
        }
        if not referenced_evidence.issubset(evidence_ids):
            return ForecastAgentAbstained(
                reason=ForecastAbstentionReason.UNKNOWN_EVIDENCE_REFERENCE,
                detail="provider scenario references unadmitted evidence",
            )
        historical_ids = {
            episode.dag_id for episode in forecast_input.historical_memory
        }
        referenced_history = {
            reference.dag_id
            for scenario in result.scenarios
            for reference in scenario.historical_dag_references
        }
        if not referenced_history.issubset(historical_ids):
            return ForecastAgentAbstained(
                reason=ForecastAbstentionReason.UNKNOWN_HISTORICAL_REFERENCE,
                detail="provider scenario references unselected historical memory",
            )
        return ForecastAgentSucceeded(forecast_result=result)
