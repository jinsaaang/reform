from __future__ import annotations

from hgf.exemplar_selection import choose_top_k_ranked


def _row(rank: int, memory_id: str, evidence_count: int) -> dict:
    return {
        "rank": rank,
        "memory_question_id": memory_id,
        "historical_forecast_time_evidence_count": evidence_count,
    }


def test_top_k_preserves_v22_evidence_floor_for_first_match() -> None:
    ranked = [
        _row(1, "low-evidence-best-score", 1),
        _row(2, "first-eligible", 3),
        _row(3, "second-eligible", 2),
        _row(4, "zero-evidence", 0),
    ]
    selected = choose_top_k_ranked(ranked, top_k=3)
    assert [row["memory_question_id"] for row in selected] == [
        "first-eligible",
        "second-eligible",
        "low-evidence-best-score",
    ]
    assert [row["rank"] for row in selected] == [1, 2, 3]
    assert [row["score_rank"] for row in selected] == [2, 3, 1]
    assert selected[0]["selection_reason"] == "v22_evidence_floor"


def test_top_k_preserves_v22_rank_one_fallback() -> None:
    ranked = [
        _row(1, "fallback", 0),
        _row(2, "also-low", 1),
        _row(3, "also-zero", 0),
        _row(4, "still-low", 1),
        _row(5, "still-zero", 0),
        _row(6, "later-eligible", 5),
    ]
    selected = choose_top_k_ranked(ranked, top_k=2)
    assert [row["memory_question_id"] for row in selected] == [
        "fallback",
        "later-eligible",
    ]
    assert selected[0]["selection_reason"] == "v22_rank_one_fallback"


def test_top_k_requires_a_positive_k() -> None:
    try:
        choose_top_k_ranked([_row(1, "memory", 2)], top_k=0)
    except ValueError as exc:
        assert "top_k must be positive" in str(exc)
    else:
        raise AssertionError("top_k=0 must fail")
