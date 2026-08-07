"""Run frozen Procedural Topology HGF with an exact provider route.

The forecasting prompts, schemas, validators, and stage order are unchanged.
Only the OpenRouter provider policy is added to each request so that one exact
endpoint family handles the complete run and every requested parameter must be
supported.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from hgf.forecast_core import _atomic_write
from hgf_e2e_topology_sidecar import run as sidecar_run


def _parse_policy_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider-only", required=True)
    parser.add_argument("--disable-native-reasoning", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def _provider_policy(provider_only: str) -> dict[str, Any]:
    if not provider_only.strip():
        raise ValueError("provider tag must not be empty")
    return {
        "only": [provider_only],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def _with_provider_policy(
    kwargs: dict[str, Any],
    provider_only: str,
    *,
    disable_native_reasoning: bool = False,
) -> dict[str, Any]:
    forwarded = dict(kwargs)
    extra_body = copy.deepcopy(forwarded.get("extra_body") or {})
    if "provider" in extra_body:
        raise ValueError("request already contains a provider policy")
    if disable_native_reasoning:
        extra_body.pop("reasoning", None)
    extra_body["provider"] = _provider_policy(provider_only)
    forwarded["extra_body"] = extra_body
    return forwarded


class _PinnedCompletionsProxy(sidecar_run._CompletionsProxy):
    def __init__(
        self,
        target: Any,
        *,
        provider_only: str,
        disable_native_reasoning: bool = False,
    ) -> None:
        super().__init__(target)
        self._provider_only = provider_only
        self._disable_native_reasoning = disable_native_reasoning

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return super().create(
            *args,
            **_with_provider_policy(
                kwargs,
                self._provider_only,
                disable_native_reasoning=self._disable_native_reasoning,
            ),
        )


class _PinnedChatProxy:
    def __init__(
        self,
        target: Any,
        *,
        provider_only: str,
        disable_native_reasoning: bool,
    ) -> None:
        self._target = target
        self.completions = _PinnedCompletionsProxy(
            target.completions,
            provider_only=provider_only,
            disable_native_reasoning=disable_native_reasoning,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class PinnedRecordingClientProxy:
    provider_only = ""
    disable_native_reasoning = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._target = sidecar_run.OriginalOpenAI(*args, **kwargs)
        self.chat = _PinnedChatProxy(
            self._target.chat,
            provider_only=self.provider_only,
            disable_native_reasoning=self.disable_native_reasoning,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _amend_manifests(
    output_dir: Path,
    *,
    provider_only: str,
    disable_native_reasoning: bool,
) -> None:
    policy = _provider_policy(provider_only)
    sidecar_path = output_dir / "sidecar_manifest.json"
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["request_forwarded_unchanged"] = False
        sidecar["request_modified_by_execution_policy_only"] = True
        sidecar["provider_policy"] = policy
        sidecar["native_reasoning_parameter_forwarded"] = not disable_native_reasoning
        _atomic_write(sidecar_path, sidecar)

    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["provider_routing"] = {
            **policy,
            "response_generation_id_required": True,
            "endpoint_id_enrichment_required": True,
        }
        protocol["execution_native_reasoning_effort"] = (
            "none" if disable_native_reasoning else "requested"
        )
        _atomic_write(protocol_path, protocol)

    _atomic_write(
        output_dir / "provider_policy_manifest.json",
        {
            "schema_version": "hgf_provider_policy_manifest_v1",
            "provider_policy": policy,
            "forecast_method_modified": False,
            "request_modified_by_execution_policy_only": True,
            "endpoint_id_enrichment_required": True,
            "native_reasoning_parameter_forwarded": not disable_native_reasoning,
        },
    )


def main() -> None:
    args = _parse_policy_args()
    output_dir = sidecar_run._output_dir_from_argv()
    PinnedRecordingClientProxy.provider_only = args.provider_only
    PinnedRecordingClientProxy.disable_native_reasoning = args.disable_native_reasoning
    frozen_run = sidecar_run.frozen_run
    frozen_run.OpenAI = PinnedRecordingClientProxy
    frozen_run._run_case = sidecar_run._recording_run_case
    completed = False
    sidecar_run._write_manifest(output_dir, completed=False)
    _amend_manifests(
        output_dir,
        provider_only=args.provider_only,
        disable_native_reasoning=args.disable_native_reasoning,
    )
    try:
        frozen_run.main()
        completed = True
    finally:
        sidecar_run._write_manifest(output_dir, completed=completed)
        _amend_manifests(
            output_dir,
            provider_only=args.provider_only,
            disable_native_reasoning=args.disable_native_reasoning,
        )


if __name__ == "__main__":
    main()
