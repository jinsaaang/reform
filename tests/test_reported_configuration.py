from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dags
import build_hgf_artifacts
import run_fresh_pipeline
import run_live_forecast
from hgf.boundary import _WEAK_MAGNITUDE_CONFIDENCE_CAP
from hgf_e2e_topology import run as reform_run


def test_dag_defaults_match_reported_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_dags", "--memory-questions", "memory.jsonl", "--output-root", "out"],
    )
    args = build_dags._args()
    assert args.model == "google/gemini-2.5-flash"
    assert args.min_evidence_articles == 10
    assert args.min_graph_events == 8
    assert args.min_graph_depth == 3
    assert args.search_query_budget == 10
    assert args.max_evidence_rounds == 3


def test_reform_defaults_match_reported_inference(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["reform"])
    args = reform_run._parse_args()
    assert args.workers == 20
    assert args.candidate_evidence_limit == 80
    assert args.evidence_limit == 20
    assert args.forecast_evidence_limit == 14
    assert args.max_dags == 3
    assert args.max_paths == 3
    assert args.max_checkpoints == 12
    assert args.ledger_max_tokens == 4000
    assert args.graph_max_tokens == 5000
    assert args.reasoning_max_tokens == 5000
    assert args.boundary_max_tokens == 2400
    assert args.reasoning_effort == "medium"
    assert args.max_output_tokens == 16000
    assert _WEAK_MAGNITUDE_CONFIDENCE_CAP == {
        "direction_only": 0.50,
        "insufficient": 0.45,
    }


def test_public_workflow_uses_reported_models_and_parallelism(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fresh",
            "--memory-questions",
            "memory.jsonl",
            "--test-questions",
            "test.jsonl",
            "--work-dir",
            "out",
        ],
    )
    fresh = run_fresh_pipeline._args()
    assert fresh.dag_model == "google/gemini-2.5-flash"
    assert fresh.model == "google/gemini-2.5-flash-lite"
    assert fresh.dag_workers == 2
    assert fresh.exemplar_workers == 10
    assert fresh.forecast_workers == 20

    monkeypatch.setattr(sys, "argv", ["artifacts", "--artifact-root", "out"])
    artifacts = build_hgf_artifacts._args()
    assert artifacts.model == "google/gemini-2.5-flash-lite"
    assert artifacts.workers == 10

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecast",
            "--artifact-root",
            "out",
            "--test-questions",
            "test.jsonl",
            "--output-dir",
            "forecast-out",
        ],
    )
    forecast = run_live_forecast._args()
    assert forecast.model == "google/gemini-2.5-flash-lite"
    assert forecast.workers == 20
    assert forecast.reasoning_effort == "medium"
    assert forecast.max_output_tokens == 16000
