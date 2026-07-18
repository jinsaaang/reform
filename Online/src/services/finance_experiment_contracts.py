"""Provider contracts and deterministic seeds for finance experiment batches."""

import hashlib
from dataclasses import dataclass
from typing import Final, Protocol

from src.agents.finance_reasoning_judge import JudgeProvider
from src.agents.finance_search_agent import SearchProvider
from src.domain.finance.experiment_manifest import (
    FinanceExperimentManifest,
    SafeCompletionSettings,
    Seed,
)
from src.domain.finance.experiment_pairs import ForecastPair, PAIR_ARMS
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.memory import QuestionId
from src.integrations.finance_live_forecast import (
    TransientForecastProviderCompletion,
)

_SEED_DOMAIN: Final = "finance-experiment-seed/v1"
_SEED_MASK: Final = (1 << 63) - 1


class ExperimentForecastProvider(Protocol):
    """One-attempt forecast completion with safe telemetry."""

    def complete(
        self,
        forecast_input: ForecasterInput,
    ) -> TransientForecastProviderCompletion: ...


class ForecastProviderFactory(Protocol):
    """Construct a forecast provider from exact typed settings."""

    def __call__(
        self,
        settings: SafeCompletionSettings,
    ) -> ExperimentForecastProvider: ...


class JudgeProviderFactory(Protocol):
    """Construct a judge provider from exact typed settings."""

    def __call__(self, settings: SafeCompletionSettings) -> JudgeProvider: ...


class FinanceExperimentProviderBuilder(Protocol):
    """Construct all external providers after service preflight."""

    def __call__(
        self,
        manifest: FinanceExperimentManifest,
    ) -> "FinanceExperimentProviders": ...


@dataclass(frozen=True, slots=True)
class FinanceExperimentProviderBuildError(Exception):
    """Provider construction rejected a validated manifest."""

    detail: str

    def __str__(self) -> str:
        return f"finance experiment provider construction failed: {self.detail}"


@dataclass(frozen=True, slots=True)
class FinanceExperimentProviders:
    """External provider capabilities constructed after service preflight."""

    search_provider: SearchProvider
    forecast_provider_factory: ForecastProviderFactory
    judge_provider_factory: JudgeProviderFactory


@dataclass(frozen=True, slots=True)
class FinanceTrialSeedInputs:
    """Stable coordinates shared by all random choices in one trial."""

    root_seed: Seed
    question_id: QuestionId
    repetition_index: int


@dataclass(frozen=True, slots=True)
class _SeedMaterial:
    trial: FinanceTrialSeedInputs
    purpose: str
    member_id: str
    order_index: int = 0


def _derive_seed(material: _SeedMaterial) -> Seed:
    trial = material.trial
    encoded = (
        f"{_SEED_DOMAIN}\0{trial.root_seed}\0{trial.question_id}\0"
        f"{trial.repetition_index}\0{material.purpose}\0"
        f"{material.member_id}\0{material.order_index}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return value & _SEED_MASK


def derive_forecast_seed(inputs: FinanceTrialSeedInputs) -> Seed:
    """Derive the common A/B/C forecast provider seed."""
    return _derive_seed(_SeedMaterial(inputs, "forecast", "-"))


def derive_panel_seed(
    inputs: FinanceTrialSeedInputs,
    pair: ForecastPair,
) -> Seed:
    """Derive one pair's local member/orientation scheduling seed."""
    first, second = PAIR_ARMS[pair]
    purpose = f"panel-order:{first.value}:{second.value}"
    return _derive_seed(_SeedMaterial(inputs, purpose, "-"))


def derive_judge_seed(
    inputs: FinanceTrialSeedInputs,
    pair: ForecastPair,
    member_id: str,
) -> Seed:
    """Derive one member/pair seed shared by both answer orientations."""
    first, second = PAIR_ARMS[pair]
    purpose = f"judge:{first.value}:{second.value}:{member_id}"
    return _derive_seed(_SeedMaterial(inputs, purpose, member_id))


__all__ = [
    "ExperimentForecastProvider",
    "FinanceExperimentProviderBuildError",
    "FinanceExperimentProviderBuilder",
    "FinanceExperimentProviders",
    "FinanceTrialSeedInputs",
    "ForecastProviderFactory",
    "JudgeProviderFactory",
    "derive_forecast_seed",
    "derive_judge_seed",
    "derive_panel_seed",
]
