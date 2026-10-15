from datetime import datetime, timezone
from types import SimpleNamespace

from src.pipelines.prompts.graph_builder import get_prompt, get_repair_prompt


def _question():
    return SimpleNamespace(
        id="q_mcq",
        question_text="Which interval will contain the monthly return?",
        resolution_date=datetime(2025, 1, 31, tzinfo=timezone.utc),
        ground_truth="Between -2% and 2%",
        causal_explanation="Demand weakened while supply remained ample.",
    )


def test_graph_prompt_supports_all_mcq_outcomes():
    prompt = get_prompt(
        _question(),
        actual_outcome_event_id="evt_actual",
        min_graph_depth=3,
        min_events=8,
    )

    assert "for EVERY outcome event" in prompt
    assert "For MCQ questions with more than two options" in prompt
    assert "import json" in prompt
    assert "record_outcome_impact` stores metadata only" in prompt
    assert "Never submit the same" in prompt
    assert "Do NOT create a proxy event" in prompt
    assert "never index an alias map with a" in prompt


def test_repair_prompt_preserves_valid_graph_content():
    prompt = get_repair_prompt(
        question=_question(),
        actual_outcome_event_id="evt_actual",
        issues=["Actual outcome event is orphaned."],
        min_graph_depth=3,
        min_events=8,
    )

    compact = " ".join(prompt.split())
    assert "Never clear, replace, or rebuild the graph wholesale" in compact
    assert "Always delete an uncited proxy" in compact
    assert "never recreate a report merely to repair chronology or an edge" in compact
    assert "exact raw event ID returned by graph_inspector" in compact
    assert "add the missing final incoming edge" in compact
    assert "without deleting valid graph content" in compact
    assert "call delete_event on that exact ID first" in compact
