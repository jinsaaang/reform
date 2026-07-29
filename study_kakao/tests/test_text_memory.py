from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from hgf import text_memory


def test_distilled_text_memory_is_written_under_a_local_lock(
    monkeypatch,
    tmp_path,
) -> None:
    question = SimpleNamespace(
        id="memory-question",
        ground_truth="within",
        question_text="Historical question",
        context="Historical context",
        resolution_reasoning="Historical resolution",
    )
    memory = {
        "target_type": "monthly change",
        "evidence_priorities": ["official release"],
        "reasoning_guidance": ["compare current drivers"],
        "counterevidence_guidance": ["check reversal signals"],
        "calibration_guidance": ["retain uncertainty"],
    }
    written = []

    monkeypatch.setattr(
        text_memory,
        "resolve_forecast_cutoff",
        lambda _: (datetime(2025, 1, 1, tzinfo=timezone.utc), "test"),
    )
    monkeypatch.setattr(
        text_memory,
        "_exemplar_article_ids",
        lambda *_: set(),
    )
    monkeypatch.setattr(text_memory, "_target_contract", lambda _: {})
    monkeypatch.setattr(
        text_memory,
        "_call_with_repair",
        lambda *args, **kwargs: (memory, {}, {}, 0.0, False),
    )
    monkeypatch.setattr(
        text_memory,
        "_atomic_write",
        lambda path, payload: written.append((path, payload)),
    )

    result = text_memory._distill_text_memory(
        client=object(),
        model="test-model",
        memory_question=question,
        memory_graph={"evidence": {"articles": []}},
        cache_dir=tmp_path,
        max_tokens=100,
    )

    assert result["memory"] == memory
    assert written == [(tmp_path / "memory-question.json", memory)]
