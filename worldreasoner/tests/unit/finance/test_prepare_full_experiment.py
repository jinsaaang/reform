from datetime import datetime, timedelta, timezone

from forecaster.data_pipeline.family import resolve_forecast_cutoff
from forecaster.experiments.prepare_full import build_full_split
from src.domain.models import Question


def _question(index: int) -> Question:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    return Question(
        id=f"q{index:03d}",
        question_text=f"Will synthetic financial outcome {index} occur?",
        question_type="binary",
        domain="finance",
        source="test",
        difficulty=3,
        estimated_start_time=start,
        resolution_date=start + timedelta(days=10),
        ground_truth=True,
        options=["yes", "no"],
    )


def test_full_split_is_chronological_and_memory_is_cutoff_safe(tmp_path):
    input_path = tmp_path / "questions.jsonl"
    questions = [_question(index) for index in range(6)]
    input_path.write_text(
        "".join(question.model_dump_json() + "\n" for question in reversed(questions)),
        encoding="utf-8",
    )

    manifest = build_full_split(
        input_path,
        tmp_path / "experiment",
        memory_count=4,
        test_count=2,
    )

    assert manifest["memory_question_ids"] == ["q000", "q001", "q002", "q003"]
    assert [row["question_id"] for row in manifest["test"]] == ["q004", "q005"]
    for row in manifest["test"]:
        cutoff = datetime.fromisoformat(row["forecast_cutoff"])
        eligible = set(row["eligible_memory_question_ids"])
        for question in questions[:4]:
            assert (question.id in eligible) == (question.resolution_date < cutoff)


def test_explicit_as_of_date_overrides_late_slot():
    question = Question(
        id="corp_intc_2026_q1_revenue_bucket",
        question_text=(
            "As of 2026-03-28, which bucket will INTEL CORP report for revenue?"
        ),
        question_type="mcq",
        domain="finance",
        source="test",
        difficulty=3,
        estimated_start_time=datetime(2026, 3, 28, tzinfo=timezone.utc),
        resolution_date=datetime(2026, 4, 24, 23, 59, 59, tzinfo=timezone.utc),
        ground_truth="$13B-$14B",
        options=["below $13B", "$13B-$14B", "at least $14B"],
        metadata={
            "finfactorbench": {"forecast_date_options": ["2026-03-28"]}
        },
    )

    cutoff, source = resolve_forecast_cutoff(question)

    assert cutoff == datetime(2026, 3, 28, tzinfo=timezone.utc)
    assert source == "metadata.forecast_date_options"


def test_full_split_respects_family_preassignment(tmp_path):
    questions = [_question(index) for index in range(6)]
    # Make the globally earliest question a test item to prove that source
    # preassignment, rather than global sorting, controls the split.
    assignments = ["test", "memory", "memory", "memory", "memory", "test"]
    for question, split in zip(questions, assignments):
        question.metadata = {
            "finance": {
                "split": split,
                "event_cluster_id": f"cluster-{question.id}",
                "forecast_date_options": [
                    question.estimated_start_time.date().isoformat()
                ],
            }
        }
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        "".join(question.model_dump_json() + "\n" for question in questions),
        encoding="utf-8",
    )

    manifest = build_full_split(
        input_path,
        tmp_path / "experiment",
        memory_count=4,
        test_count=2,
    )

    assert manifest["selection_policy"]["split"] == (
        "dataset_preassigned_family_chronological_split"
    )
    assert manifest["memory_question_ids"] == ["q001", "q002", "q003", "q004"]
    assert [row["question_id"] for row in manifest["test"]] == ["q000", "q005"]


def test_full_split_can_filter_active_finance_category(tmp_path):
    questions = [_question(index) for index in range(8)]
    for index, question in enumerate(questions):
        category = "macro" if index < 6 else "market_fx_credit"
        split = "memory" if index < 4 else "test"
        if category != "macro":
            split = "memory"
        question.metadata = {
            "finance": {
                "category": category,
                "split": split,
                "event_cluster_id": f"cluster-{question.id}",
                "forecast_date_options": [question.estimated_start_time.date().isoformat()],
            }
        }
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        "".join(question.model_dump_json() + "\n" for question in questions),
        encoding="utf-8",
    )

    manifest = build_full_split(
        input_path,
        tmp_path / "experiment",
        memory_count=4,
        test_count=2,
        category="macro",
    )

    assert manifest["category"] == "macro"
    assert manifest["question_count"] == 6
    assert set(manifest["memory_question_ids"]) == {"q000", "q001", "q002", "q003"}
