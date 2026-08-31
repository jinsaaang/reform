from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_fresh_pipeline as fresh
from hgf.live_evidence import build_live_queries
from hgf_e2e_topology_live.run import _aggregate


def _question(question_id: str, *, split: str, ground_truth: str | None) -> dict:
    return {
        "id": question_id,
        "question_text": "Which range will quarterly revenue growth enter?",
        "question_type": "mcq",
        "domain": "finance",
        "source": "test",
        "difficulty": 1,
        "resolution_date": (
            "2024-10-20T00:00:00Z" if split == "memory" else "2025-04-20T00:00:00Z"
        ),
        "estimated_start_time": "2025-03-31T00:00:00Z",
        "ground_truth": ground_truth,
        "options": ["below", "within", "above"],
        "metadata": {
            "finance": {
                "family_id": "issuer_revenue",
                "target_metric": "quarterly revenue growth acceleration",
                "target_period": "fiscal quarter ending 2025-03-31",
                "forecast_cutoff": "2025-03-31T00:00:00Z",
                "category": "corporate_earnings",
                "entity": "Example Corp. (EXM)",
                "split": split,
            }
        },
    }


def test_fresh_workspace_selects_cutoff_eligible_history(tmp_path: Path) -> None:
    memory = _question("memory_1", split="memory", ground_truth="above")
    target = _question("target_1", split="test", ground_truth=None)
    fresh._prepare_question_workspace(
        work_dir=tmp_path,
        memory_rows=[memory],
        test_rows=[target],
    )
    fixed = json.loads(
        (tmp_path / "data/memory_bank/fixed_exemplar_selection.json").read_text()
    )
    assert fixed["entries"] == [
        {"question_id": "target_1", "memory_question_id": "memory_1"}
    ]
    assert (tmp_path / "configs/reproduction.json").is_file()


def test_live_queries_use_factors_but_not_relations_or_answers() -> None:
    target = _question("target_1", split="test", ground_truth=None)
    question = SimpleNamespace(
        question_text=target["question_text"], metadata=target["metadata"]
    )
    blueprint = {
        "search_factors": [{"factor": "[CURRENT_PERIOD_REQUIRED] deliveries"}],
        "topology": {"edges": [{"relationship": "causes-secret-relation"}]},
    }
    queries = build_live_queries(
        question,
        [blueprint],
        datetime(2025, 3, 31, tzinfo=UTC),
        limit=4,
    )
    rendered = " ".join(queries)
    assert "Q1 2025" in rendered
    assert "deliveries" in rendered
    assert "causes-secret-relation" not in rendered
    assert "CURRENT_PERIOD_REQUIRED" not in rendered
    assert all(query.endswith("before:2025-03-31") for query in queries)


def test_unresolved_forecasts_do_not_break_aggregate() -> None:
    result = _aggregate(
        [{"status": "success", "category": "macro", "metrics": None}]
    )
    assert result["success_count"] == 1
    assert result["scored_count"] == 0
    assert result["overall"] == {}
