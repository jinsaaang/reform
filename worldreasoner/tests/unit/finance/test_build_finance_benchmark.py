import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.domain.models import Question


DATA_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "worldreasoner"
    / "finance_questions_500.jsonl"
)
HGF_DATA_PATH = DATA_PATH.with_name("finance_questions_hgf_300.jsonl")


def test_long_horizon_benchmark_is_balanced_and_family_recurrent():
    questions = [
        Question.model_validate_json(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories = Counter()
    splits = Counter()
    question_types = Counter()
    category_types = defaultdict(Counter)
    families = Counter()
    family_types = {}
    labels = defaultdict(Counter)

    for question in questions:
        metadata = question.metadata["finance"]
        private = question.metadata["benchmark_private"]
        categories[metadata["category"]] += 1
        splits[metadata["split"]] += 1
        question_types[question.question_type.value] += 1
        category_types[metadata["category"]][question.question_type.value] += 1
        families[metadata["family_id"]] += 1
        family_types[metadata["family_id"]] = question.question_type.value
        labels[metadata["family_id"]][private["resolved_outcome_label"]] += 1
        assert "resolution_value" not in metadata
        assert question.estimated_start_time < question.resolution_date
        if question.question_type.value == "mcq":
            assert question.options == [
                "below recent range",
                "within recent range",
                "above recent range",
            ]
            assert question.ground_truth in question.options
            thresholds = metadata["comparison_thresholds"]
            assert thresholds["lower"] < thresholds["upper"]
        else:
            assert question.question_type.value == "binary"
            assert isinstance(question.ground_truth, bool)
            assert private["resolved_outcome_label"] in {"yes", "no"}

    assert len(questions) == 500
    assert categories == {
        "corporate_earnings": 100,
        "energy_commodities": 100,
        "macro": 100,
        "market_fx_credit": 100,
        "monetary_policy": 100,
    }
    assert splits == {"memory": 340, "test": 160}
    assert question_types == {"mcq": 315, "binary": 185}
    assert all(
        counts == {"mcq": 63, "binary": 37}
        for counts in category_types.values()
    )
    assert len(families) == 40
    assert set(families.values()) == {12, 13}
    for family_id, counts in labels.items():
        if family_types[family_id] == "mcq":
            assert len(counts) == 3
            assert min(counts.values()) >= 2
        else:
            assert len(counts) == 2
            assert min(counts.values()) >= 3


def test_long_horizon_benchmark_manifest_matches_dataset():
    manifest_path = DATA_PATH.with_name("finance_questions_500_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["question_count"] == 500
    assert manifest["family_count"] == 40
    assert len(manifest["memory_question_ids"]) == 340
    assert len(manifest["test_question_ids"]) == 160
    assert manifest["cluster_split_violations"] == 0
    assert manifest["question_type_counts"] == {"mcq": 315, "binary": 185}
    assert all(
        counts == {"mcq": 63, "binary": 37}
        for counts in manifest["category_type_counts"].values()
    )
    assert manifest["same_family_eligible_memory"] == {"min": 8, "max": 9}


def test_hgf_benchmark_is_balanced_and_cutoff_safe():
    questions = [
        Question.model_validate_json(line)
        for line in HGF_DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories = Counter()
    splits = Counter()
    category_types = defaultdict(Counter)
    memory_by_family = defaultdict(list)

    for question in questions:
        metadata = question.metadata["finance"]
        categories[metadata["category"]] += 1
        splits[metadata["split"]] += 1
        category_types[metadata["category"]][question.question_type.value] += 1
        assert question.estimated_start_time >= datetime(
            2023, 1, 1, tzinfo=timezone.utc
        )
        assert question.estimated_start_time < question.resolution_date
        if metadata["split"] == "memory":
            memory_by_family[metadata["family_id"]].append(question)

    assert len(questions) == 300
    assert set(categories.values()) == {60}
    assert splits == {"memory": 200, "test": 100}
    assert all(
        counts == {"mcq": 38, "binary": 22}
        for counts in category_types.values()
    )
    assert len(memory_by_family) == 40
    assert all(len(items) == 5 for items in memory_by_family.values())

    for question in questions:
        metadata = question.metadata["finance"]
        if metadata["split"] != "test":
            continue
        assert question.estimated_start_time >= datetime(
            2025, 2, 1, tzinfo=timezone.utc
        )
        eligible = [
            memory
            for memory in memory_by_family[metadata["family_id"]]
            if memory.resolution_date < question.estimated_start_time
        ]
        assert len(eligible) == 5
