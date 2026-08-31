from types import SimpleNamespace

from src.analysis.forecast_graph_validation import validate_forecast_graph


def _event(event_id: str, session_id: str = "session"):
    return SimpleNamespace(id=event_id, session_id=session_id)


def _hypothesis(
    hypothesis_id: str,
    source: str,
    target: str,
    *,
    session_id: str = "session",
    evidence: tuple[str, ...] = ("article",),
):
    return SimpleNamespace(
        id=hypothesis_id,
        session_id=session_id,
        source_event_id=source,
        target_event_id=target,
        evidence_article_ids=list(evidence),
    )


def test_valid_forecast_dag_passes() -> None:
    result = validate_forecast_graph(
        [_event("driver"), _event("target")],
        [_hypothesis("edge", "driver", "target")],
    )

    assert result.is_valid
    assert result.to_dict() == {
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }


def test_missing_endpoint_and_mixed_sessions_fail() -> None:
    result = validate_forecast_graph(
        [_event("driver")],
        [_hypothesis("edge", "driver", "missing", session_id="other")],
    )

    assert not result.is_valid
    assert any("missing events" in error for error in result.errors)
    assert any("mixes sessions" in error for error in result.errors)


def test_self_loop_and_cycle_fail() -> None:
    result = validate_forecast_graph(
        [_event("a"), _event("b")],
        [
            _hypothesis("self", "a", "a"),
            _hypothesis("forward", "a", "b"),
            _hypothesis("back", "b", "a"),
        ],
    )

    assert not result.is_valid
    assert any("self-loop" in error for error in result.errors)
    assert any("Cycle detected" in error for error in result.errors)


def test_missing_evidence_and_isolated_events_warn() -> None:
    result = validate_forecast_graph(
        [_event("driver"), _event("target"), _event("isolated")],
        [_hypothesis("edge", "driver", "target", evidence=())],
    )

    assert result.is_valid
    assert any("no supporting evidence" in warning for warning in result.warnings)
    assert any("Isolated forecast events" in warning for warning in result.warnings)
