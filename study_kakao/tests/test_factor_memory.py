from __future__ import annotations

from hgf import baselines


def test_factor_memory_has_an_explicit_prompt_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call_with_repair(*args, **kwargs):
        captured.update(kwargs)
        return {}, {}, {}, 0.0, False

    monkeypatch.setattr(
        baselines,
        "_call_with_repair",
        fake_call_with_repair,
    )

    baselines._call_memory_reasoning(
        client=object(),
        model="test-model",
        question_id="question-1",
        public_case={"question": "Test question"},
        evidence=[],
        options=["lower", "higher"],
        memory_type="factor",
        memory={
            "view": "hgf_search_cards",
            "instructions": "Use as coverage hints only.",
            "factors": [{"factor": "demand"}],
        },
        max_tokens=100,
    )

    prompt = str(captured["prompt"])
    assert "Factor search cards are supplied" in prompt
    assert "No historical memory is available" not in prompt
    assert '"view": "hgf_search_cards_v1"' in prompt
    assert captured["seed"] == baselines._seed(
        "question-1",
        "paper-factor-reasoning",
    )
