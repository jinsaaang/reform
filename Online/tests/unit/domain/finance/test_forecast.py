"""Tests for non-graph-building forecast contracts."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.memory import OutcomeLabel, QuestionId, ScenarioId
from src.domain.finance.search import EvidencePack, TargetProfile
from tests.unit.domain.finance._factories import make_episode


def _target() -> TargetProfile:
    return TargetProfile.model_validate(
        {
            "question_id": "current-question",
            "question_text": "Will NVIDIA revenue exceed analyst expectations?",
            "question_type": "binary",
            "domain": "finance",
            "context": ("GPU demand remains elevated",),
            "cutoff": "2025-01-01T00:00:00+00:00",
            "outcome_space": ("Yes", "No"),
            "resolution_rule": "Use the filed quarterly revenue.",
        }
    )


def _scenario(
    scenario_id: str,
    probability: float,
    yes_probability: float,
) -> Scenario:
    return Scenario(
        scenario_id=ScenarioId(scenario_id),
        name=f"Scenario {scenario_id}",
        reasoning_steps=("Apply admitted evidence.",),
        probability=probability,
        conditional_outcomes=(
            OutcomeProbability(
                label=OutcomeLabel("Yes"),
                probability=yes_probability,
            ),
            OutcomeProbability(
                label=OutcomeLabel("No"),
                probability=1.0 - yes_probability,
            ),
        ),
        evidence_ids=(),
        historical_dag_references=(),
        assumptions=(),
        triggers=(),
        disconfirmers=(),
        uncertainty="The evidence may change.",
    )


def _final_outcomes(yes_probability: float) -> tuple[OutcomeProbability, ...]:
    return (
        OutcomeProbability(
            label=OutcomeLabel("Yes"),
            probability=yes_probability,
        ),
        OutcomeProbability(
            label=OutcomeLabel("No"),
            probability=1.0 - yes_probability,
        ),
    )


class TestProbabilityBoundary:
    def test_should_reject_probability_outside_unit_interval(self) -> None:
        # Given: a provider probability greater than one
        # When: the probability crosses the forecast boundary
        with pytest.raises(ValidationError):
            _ = OutcomeProbability(label=OutcomeLabel("Yes"), probability=1.01)

        # Then: invalid provider output is rejected

    def test_should_reject_non_normalized_scenario_distribution(self) -> None:
        # Given: a scenario whose conditional outcomes sum below one
        outcomes = (
            OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.4),
            OutcomeProbability(label=OutcomeLabel("No"), probability=0.4),
        )

        # When: the scenario crosses the forecast boundary
        with pytest.raises(ValidationError):
            _ = Scenario(
                scenario_id=ScenarioId("scenario-1"),
                name="Demand persists",
                reasoning_steps=("Demand supports revenue.",),
                probability=1.0,
                conditional_outcomes=outcomes,
                evidence_ids=(),
                historical_dag_references=(),
                assumptions=("Margins remain stable.",),
                triggers=("Guidance rises.",),
                disconfirmers=("Orders fall.",),
                uncertainty="Demand can normalize.",
            )

        # Then: aggregation cannot proceed with an invalid distribution


class TestForecasterInputBoundary:
    def test_should_reject_target_question_inside_historical_memory(self) -> None:
        # Given: historical memory that carries the current target's exact ID
        episode = replace(
            make_episode(),
            question_id=QuestionId("current-question"),
        )

        # When: the memory crosses the Forecaster input boundary
        with pytest.raises(ValidationError) as error:
            _ = ForecasterInput(
                target_profile=_target(),
                evidence_pack=EvidencePack(
                    items=(),
                    historical_dag_references=(),
                ),
                historical_memory=(episode,),
            )

        # Then: current-question hindsight is rejected by a stable error type
        assert error.value.errors()[0]["type"] == "current_target_in_historical_memory"

    def test_should_accept_distinct_historical_question_id(self) -> None:
        # Given: an episode with an ID distinct from the current target
        episode = make_episode()

        # When: the memory crosses the Forecaster input boundary
        forecast_input = ForecasterInput(
            target_profile=_target(),
            evidence_pack=EvidencePack(items=(), historical_dag_references=()),
            historical_memory=(episode,),
        )

        # Then: the distinct historical episode remains available to forecast
        assert forecast_input.historical_memory == (episode,)

    def test_should_namespace_historical_outcome_without_current_hindsight(
        self,
    ) -> None:
        # Given: a sanitized target and an already-built historical episode
        forecast_input = ForecasterInput(
            target_profile=_target(),
            evidence_pack=EvidencePack(items=(), historical_dag_references=()),
            historical_memory=(make_episode(),),
        )

        # When: the provider input is serialized
        serialized = forecast_input.model_dump_json()

        # Then: only explicitly historical outcome metadata is representable
        assert (
            '"historical_outcome"' in serialized
            and '"ground_truth"' not in serialized
            and '"current_dag"' not in serialized
        )


class TestForecastResultBoundary:
    def test_should_reject_empty_scenarios(self) -> None:
        # Given: normalized final outcomes without any reasoning scenario
        # When: the provider result crosses the output boundary
        with pytest.raises(ValidationError):
            _ = ForecastResult(
                scenarios=(),
                outcome_probabilities=_final_outcomes(0.5),
                explanation="No scenarios were supplied.",
            )

        # Then: an outcome forecast cannot omit its scenario basis

    def test_should_reject_duplicate_scenario_ids(self) -> None:
        # Given: two normalized scenarios sharing one stable identity
        first = _scenario("duplicate", 0.5, 0.6)
        second = _scenario("duplicate", 0.5, 0.4)

        # When: the provider result crosses the output boundary
        with pytest.raises(ValidationError) as error:
            _ = ForecastResult(
                scenarios=(first, second),
                outcome_probabilities=_final_outcomes(0.5),
                explanation="Duplicate scenario identity.",
            )

        # Then: ambiguous scenario identity is rejected
        assert error.value.errors()[0]["type"] == "duplicate_scenario_id"

    def test_should_reject_inconsistent_scenario_outcome_labels(self) -> None:
        # Given: a normalized scenario over labels absent from final outcomes
        scenario = Scenario(
            scenario_id=ScenarioId("different-labels"),
            name="Different labels",
            reasoning_steps=("Use an incompatible outcome space.",),
            probability=1.0,
            conditional_outcomes=(
                OutcomeProbability(label=OutcomeLabel("Up"), probability=0.5),
                OutcomeProbability(label=OutcomeLabel("Down"), probability=0.5),
            ),
            evidence_ids=(),
            historical_dag_references=(),
            assumptions=(),
            triggers=(),
            disconfirmers=(),
            uncertainty="Labels may differ.",
        )

        # When: the provider result crosses the output boundary
        with pytest.raises(ValidationError) as error:
            _ = ForecastResult(
                scenarios=(scenario,),
                outcome_probabilities=_final_outcomes(0.5),
                explanation="Outcome spaces differ.",
            )

        # Then: scenario and final outcome spaces must agree
        assert error.value.errors()[0]["type"] == "scenario_outcome_labels"

    def test_should_reject_inconsistent_weighted_outcomes(self) -> None:
        # Given: scenarios whose weighted Yes probability is exactly one half
        scenarios = (
            _scenario("upside", 0.25, 0.8),
            _scenario("baseline", 0.75, 0.4),
        )

        # When: final outcomes disagree with the weighted scenario aggregation
        with pytest.raises(ValidationError) as error:
            _ = ForecastResult(
                scenarios=scenarios,
                outcome_probabilities=_final_outcomes(0.6),
                explanation="Final probabilities disagree with scenarios.",
            )

        # Then: an inconsistent final probability is rejected
        assert (
            error.value.errors()[0]["type"]
            == "inconsistent_weighted_outcome_probability"
        )

    def test_should_accept_weighted_scenario_aggregation(self) -> None:
        # Given: scenarios whose weighted Yes and No probabilities are one half
        scenarios = (
            _scenario("upside", 0.25, 0.8),
            _scenario("baseline", 0.75, 0.4),
        )

        # When: the matching final probabilities cross the output boundary
        result = ForecastResult(
            scenarios=scenarios,
            outcome_probabilities=_final_outcomes(0.5),
            explanation="Final probabilities aggregate both scenarios.",
        )

        # Then: the coherent forecast remains representable
        assert result.outcome_probabilities == _final_outcomes(0.5)

    def test_should_reject_non_normalized_final_outcomes(self) -> None:
        # Given: final forecast probabilities that do not sum to one
        outcomes = (
            OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.2),
            OutcomeProbability(label=OutcomeLabel("No"), probability=0.2),
        )

        # When: the provider result crosses the output boundary
        with pytest.raises(ValidationError) as error:
            _ = ForecastResult(
                scenarios=(_scenario("normalized", 1.0, 0.5),),
                outcome_probabilities=outcomes,
                explanation="Insufficient normalized mass.",
            )

        # Then: malformed provider output is rejected
        assert error.value.errors()[0]["type"] == "outcome_probability_sum"
