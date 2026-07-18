"""Strict one-attempt LiteLLM adapter for blind finance judging."""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Final

from src.agents.finance_reasoning_judge import (
    JudgeProvider,
    JudgeProviderError,
    TransientJudgeProviderCompletion,
)
from src.core.structured_completion import (
    ChatMessage,
    StructuredCompletionClient,
    strict_response_format,
)
from src.domain.finance.experiment_manifest import SafeCompletionSettings, Seed
from src.domain.finance.judge_views import BlindJudgePayload
from src.domain.finance.judging import JudgeProviderResponse
from src.integrations.finance_completion import (
    CompletionAttemptFailed,
    CompletionAttemptRequest,
    ProviderSeedSupport,
    build_experiment_litellm_client,
    run_completion_attempt,
)

_JUDGE_INSTRUCTION: Final = """Compare the two anonymized finance forecasts.
Return one overall preference and exactly one assessment for every requested
diagnostic dimension. Use only the supplied target, evidence, memory, and
candidate reasoning. Return only the requested JSON object."""


@dataclass(frozen=True, slots=True)
class LiveJudgeProvider(JudgeProvider):
    """Call one no-tool structured completion per ordered blind payload."""

    settings: SafeCompletionSettings
    seed_support: ProviderSeedSupport
    client: StructuredCompletionClient | None = None
    clock: Callable[[], float] = monotonic

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        """Return typed telemetry or translate one safe adapter failure."""
        serialized_input = payload.model_dump_json()
        client = self.client or build_experiment_litellm_client(
            self.settings,
            requested_seed,
            self.seed_support,
        )
        messages: tuple[ChatMessage, ...] = (
            {"role": "system", "content": _JUDGE_INSTRUCTION},
            {"role": "user", "content": serialized_input},
        )
        result = run_completion_attempt(
            CompletionAttemptRequest(
                client=client,
                messages=messages,
                response_format=strict_response_format(JudgeProviderResponse),
                input_bytes=len(serialized_input.encode("utf-8")),
                requested_seed=requested_seed,
                seed_support=self.seed_support,
                provider_name=f"litellm:{self.settings.model}",
                clock=self.clock,
            )
        )
        if isinstance(result, CompletionAttemptFailed):
            raise JudgeProviderError(attempt=result.attempt)
        return TransientJudgeProviderCompletion(
            serialized_response=result.completion.content,
            attempt=result.attempt,
        )


__all__ = ["LiveJudgeProvider"]
