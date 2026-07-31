import copy
import json
from pathlib import Path

from hgf.exemplar_generator import _exemplar_schema, _validate_exemplar


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    ROOT
    / "artifacts"
    / "hgf"
    / "exemplars"
    / "memory"
    / "v3_aapl_revenue_growth_acceleration_2023_04_01.json"
)


def test_restored_schema_matches_fixed_artifact_fields() -> None:
    schema = _exemplar_schema()["schema"]
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))["worked_exemplar"]

    assert set(schema["required"]) == set(payload)


def test_restored_validator_accepts_frozen_exemplar() -> None:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))["worked_exemplar"]
    allowed = {
        str(item["article_id"])
        for item in payload["forecast_time_evidence"]
    }

    assert _validate_exemplar(payload, allowed_article_ids=allowed) == []


def test_restored_validator_rejects_unknown_evidence() -> None:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))["worked_exemplar"]
    payload = copy.deepcopy(payload)
    payload["forecast_time_evidence"].append(
        {
            "article_id": "art_post_cutoff",
            "takeaway": "Unavailable at the cutoff.",
            "why_predictive": "It must not be used.",
        }
    )

    errors = _validate_exemplar(payload, allowed_article_ids=set())

    assert any("post-cutoff/unknown evidence" in error for error in errors)
