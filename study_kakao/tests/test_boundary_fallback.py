from __future__ import annotations

from hgf.boundary import (
    _neutral_boundary_payload,
    _validate_boundary_forecast,
)


def test_neutral_boundary_fallback_passes_three_way_contract_validation() -> None:
    options = [
        "below recent range",
        "within recent range",
        "above recent range",
    ]
    contract = {
        "target_metric": "monthly change",
        "target_period": "2025-10",
        "change_unit": "percent",
        "intervals": {
            "below recent range": "(-infinity, -1.0)",
            "within recent range": "[-1.0, 1.0)",
            "above recent range": "[1.0, +infinity)",
        },
    }

    payload = _neutral_boundary_payload(options=options, contract=contract)
    probabilities, errors = _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids=set(),
        reasoning_policy="boundary_only",
        validation_policy="recovery",
    )

    assert errors == []
    assert payload["mapped_option"] == "within recent range"
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert payload["generation_fallback"].startswith("neutral_boundary")


def test_boundary_cannot_claim_more_magnitude_than_prospective_anchor() -> None:
    options = ["no", "yes"]
    contract = {
        "target_metric": "monthly change",
        "target_period": "2025-10",
        "change_unit": "percent",
        "predicate": {
            "operator": ">=",
            "threshold": 0.2,
            "yes_interval": "[0.2, +infinity)",
            "no_interval": "(-infinity, 0.2)",
        },
    }
    payload = {
        "target_operation_check": "monthly change >= 0.2 percent",
        "directional_signal": "down",
        "magnitude_assessment": {
            "support": "derived",
            "evidence_ids": ["ev1"],
            "rationale": "Prior-month direction only.",
        },
        "latent_target_estimate": {
            "low": 0.05,
            "central": 0.1,
            "high": 0.15,
            "unit": "percent",
            "basis": "Prior-month direction only.",
        },
        "boundary_checks": [
            {
                "option": "no",
                "interval": "(-infinity, 0.2)",
                "compatibility": "most_supported",
                "rationale": "Central estimate is below the threshold.",
            },
            {
                "option": "yes",
                "interval": "[0.2, +infinity)",
                "compatibility": "unsupported",
                "rationale": "Central estimate is below the threshold.",
            },
        ],
        "mapped_option": "no",
        "prediction": "no",
        "option_probabilities": [
            {"option": "no", "probability": 0.7},
            {"option": "yes", "probability": 0.3},
        ],
        "uncertainty": "High.",
    }

    _, errors = _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids={"ev1"},
        reasoning_policy="boundary_only",
        validation_policy="strict",
        prospective_anchor_support="none",
    )

    assert any("exceeds prospective target anchor" in error for error in errors)


def test_recovery_validation_never_reorders_model_probabilities() -> None:
    options = ["no", "yes"]
    contract = {
        "target_metric": "monthly change",
        "target_period": "2025-10",
        "change_unit": "percent",
        "predicate": {
            "operator": ">=",
            "threshold": 0.2,
            "yes_interval": "[0.2, +infinity)",
            "no_interval": "(-infinity, 0.2)",
        },
    }
    payload = {
        "target_operation_check": "monthly change >= 0.2 percent",
        "directional_signal": "uncertain",
        "magnitude_assessment": {
            "support": "insufficient",
            "evidence_ids": [],
            "rationale": "No exact magnitude evidence.",
        },
        "latent_target_estimate": {
            "low": -0.1,
            "central": 0.1,
            "high": 0.3,
            "unit": "percent",
            "basis": "Broad uncertainty.",
        },
        "boundary_checks": [
            {
                "option": "no",
                "interval": "(-infinity, 0.2)",
                "compatibility": "most_supported",
                "rationale": "Central estimate is below the threshold.",
            },
            {
                "option": "yes",
                "interval": "[0.2, +infinity)",
                "compatibility": "possible",
                "rationale": "Upper uncertainty crosses the threshold.",
            },
        ],
        "mapped_option": "yes",
        "prediction": "yes",
        "option_probabilities": [
            {"option": "no", "probability": 0.4},
            {"option": "yes", "probability": 0.6},
        ],
        "uncertainty": "High.",
    }

    probabilities, errors = _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids=set(),
        reasoning_policy="boundary_only",
        validation_policy="recovery",
    )

    assert probabilities == {"no": 0.4, "yes": 0.6}
    assert payload["option_probabilities"] == [
        {"option": "no", "probability": 0.4},
        {"option": "yes", "probability": 0.6},
    ]
    assert any("highest-probability" in error for error in errors)
