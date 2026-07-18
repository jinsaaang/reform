"""Explicit seed and usage telemetry for finance provider attempts."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.experiment_manifest import Seed


class _TelemetryModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ProviderSeedNotRequested(_TelemetryModel):
    status: Literal["not_requested"] = "not_requested"


class ProviderSeedEffective(_TelemetryModel):
    status: Literal["effective"] = "effective"
    requested_seed: Seed


class ProviderSeedUnsupported(_TelemetryModel):
    status: Literal["unsupported"] = "unsupported"
    requested_seed: Seed


ProviderSeedTelemetry: TypeAlias = Annotated[
    ProviderSeedNotRequested | ProviderSeedEffective | ProviderSeedUnsupported,
    Field(discriminator="status"),
]


class ProviderUsageUnavailable(_TelemetryModel):
    status: Literal["unavailable"] = "unavailable"


class ProviderUsageReported(_TelemetryModel):
    status: Literal["reported"] = "reported"
    input_tokens: Annotated[int, Field(ge=0)] | None
    output_tokens: Annotated[int, Field(ge=0)] | None
    cost_usd: Annotated[float, Field(ge=0.0)] | None

    @model_validator(mode="after")
    def validate_nonempty_usage(self) -> "ProviderUsageReported":
        """Do not claim reported usage when the provider supplied no value."""
        if all(
            value is None
            for value in (self.input_tokens, self.output_tokens, self.cost_usd)
        ):
            raise PydanticCustomError(
                "empty_provider_usage",
                "reported usage must contain at least one provider value",
            )
        return self


ProviderUsageTelemetry: TypeAlias = Annotated[
    ProviderUsageUnavailable | ProviderUsageReported,
    Field(discriminator="status"),
]


@unique
class ProviderAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"


class ProviderAttemptTelemetry(_TelemetryModel):
    """One paid-call attempt without raw request, response, or error text."""

    attempt_index: Annotated[int, Field(ge=0)]
    provider_name: str = Field(min_length=1)
    status: ProviderAttemptStatus
    latency_ms: Annotated[int, Field(ge=0)]
    input_bytes: Annotated[int, Field(ge=0)]
    output_bytes: Annotated[int, Field(ge=0)] | None
    seed: ProviderSeedTelemetry
    usage: ProviderUsageTelemetry
    failure_class: str | None = None

    @model_validator(mode="after")
    def validate_failure_surface(self) -> "ProviderAttemptTelemetry":
        """Retain only an exception class for failed or invalid attempts."""
        is_success = self.status is ProviderAttemptStatus.SUCCEEDED
        if is_success and self.failure_class is not None:
            raise PydanticCustomError(
                "success_with_failure",
                "successful attempts cannot carry a failure class",
            )
        if not is_success and not self.failure_class:
            raise PydanticCustomError(
                "failure_without_class",
                "failed attempts require a failure class",
            )
        return self


__all__ = [
    "ProviderAttemptStatus",
    "ProviderAttemptTelemetry",
    "ProviderSeedEffective",
    "ProviderSeedNotRequested",
    "ProviderSeedTelemetry",
    "ProviderSeedUnsupported",
    "ProviderUsageReported",
    "ProviderUsageTelemetry",
    "ProviderUsageUnavailable",
]
