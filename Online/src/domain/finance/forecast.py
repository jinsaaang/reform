"""Validated, non-graph-building finance Forecaster contracts."""

from typing import Annotated, ClassVar, Final, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.memory import (
    EvidenceId,
    HistoricalDagReference,
    OutcomeLabel,
    ResolvedDagEpisode,
    ScenarioId,
)
from src.domain.finance.search import EvidencePack, TargetProfile

Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
_PROBABILITY_TOLERANCE: Final = 1e-9


class OutcomeProbability(BaseModel):
    """Probability assigned to one allowed forecast outcome label."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    label: Annotated[OutcomeLabel, Field(min_length=1)]
    probability: Probability


class Scenario(BaseModel):
    """One prose reasoning branch over admitted evidence and historical memory."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    scenario_id: Annotated[ScenarioId, Field(min_length=1)]
    name: str = Field(min_length=1)
    reasoning_steps: tuple[str, ...] = Field(min_length=1)
    probability: Probability
    conditional_outcomes: tuple[OutcomeProbability, ...] = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...]
    historical_dag_references: tuple[HistoricalDagReference, ...]
    assumptions: tuple[str, ...]
    triggers: tuple[str, ...]
    disconfirmers: tuple[str, ...]
    uncertainty: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conditional_distribution(self) -> "Scenario":
        """Require unique labels and a normalized conditional distribution."""
        labels = tuple(item.label for item in self.conditional_outcomes)
        if len(labels) != len(set(labels)):
            raise PydanticCustomError(
                "duplicate_outcome_label",
                "conditional outcome labels must be unique",
            )
        total = sum(item.probability for item in self.conditional_outcomes)
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise PydanticCustomError(
                "conditional_probability_sum",
                "conditional outcome probabilities must sum to one",
            )
        return self


class ForecasterInput(BaseModel):
    """Filtered input with historical outcomes explicitly nested as memory."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    target_profile: TargetProfile
    evidence_pack: EvidencePack
    historical_memory: tuple[ResolvedDagEpisode, ...]

    @model_validator(mode="after")
    def validate_historical_references(self) -> "ForecasterInput":
        """Reject current hindsight and require references to supplied history."""
        if any(
            episode.question_id == self.target_profile.question_id
            for episode in self.historical_memory
        ):
            raise PydanticCustomError(
                "current_target_in_historical_memory",
                "historical memory must not contain the current target question",
            )
        memory_ids = tuple(episode.dag_id for episode in self.historical_memory)
        if len(memory_ids) != len(set(memory_ids)):
            raise PydanticCustomError(
                "duplicate_historical_memory",
                "historical memory DAG IDs must be unique",
            )
        referenced_ids = {
            reference.dag_id
            for reference in self.evidence_pack.historical_dag_references
        }
        if not referenced_ids.issubset(set(memory_ids)):
            raise PydanticCustomError(
                "missing_historical_memory",
                "evidence pack references unavailable historical memory",
            )
        return self


class ForecastResult(BaseModel):
    """Outcome probabilities and scenarios without a realized target outcome."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    scenarios: tuple[Scenario, ...] = Field(min_length=1)
    outcome_probabilities: tuple[OutcomeProbability, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probability_distributions(self) -> "ForecastResult":
        """Require coherent distributions within absolute tolerance 1e-9."""
        labels = tuple(item.label for item in self.outcome_probabilities)
        if len(labels) != len(set(labels)):
            raise PydanticCustomError(
                "duplicate_outcome_label",
                "forecast outcome labels must be unique",
            )
        outcome_total = sum(item.probability for item in self.outcome_probabilities)
        if abs(outcome_total - 1.0) > _PROBABILITY_TOLERANCE:
            raise PydanticCustomError(
                "outcome_probability_sum",
                "forecast outcome probabilities must sum to one",
            )
        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise PydanticCustomError(
                "duplicate_scenario_id",
                "scenario IDs must be unique",
            )
        scenario_total = sum(scenario.probability for scenario in self.scenarios)
        if abs(scenario_total - 1.0) > _PROBABILITY_TOLERANCE:
            raise PydanticCustomError(
                "scenario_probability_sum",
                "scenario probabilities must sum to one",
            )
        outcome_labels = set(labels)
        weighted_probabilities: dict[OutcomeLabel, float] = {
            label: 0.0 for label in labels
        }
        for scenario in self.scenarios:
            conditional_labels = {item.label for item in scenario.conditional_outcomes}
            if conditional_labels != outcome_labels:
                raise PydanticCustomError(
                    "scenario_outcome_labels",
                    "scenario and final outcome labels must match",
                )
            for item in scenario.conditional_outcomes:
                weighted_probabilities[item.label] += (
                    scenario.probability * item.probability
                )
        for item in self.outcome_probabilities:
            if (
                abs(weighted_probabilities[item.label] - item.probability)
                > _PROBABILITY_TOLERANCE
            ):
                raise PydanticCustomError(
                    "inconsistent_weighted_outcome_probability",
                    "final probabilities must equal weighted scenario outcomes",
                )
        return self
