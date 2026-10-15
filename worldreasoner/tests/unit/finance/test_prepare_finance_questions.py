import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "finance"
    / "prepare_finance_questions.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_finance_questions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_all_finance_records_convert_to_worldreasoner_questions():
    records = MODULE.read_jsonl(MODULE.DEFAULT_INPUT)
    questions = [MODULE.convert_record(record) for record in records]

    assert len(questions) == 300
    assert {question.domain.value for question in questions} == {"finance"}
    assert {question.source for question in questions} == {"finfactorbench"}
    assert sum(question.question_type.value == "binary" for question in questions) == 72
    assert sum(question.question_type.value == "mcq" for question in questions) == 228
    assert all(question.estimated_start_time < question.resolution_date for question in questions)
    assert all(
        question.ground_truth in question.options
        for question in questions
        if question.question_type.value == "mcq"
    )


def test_sample_has_two_questions_per_original_finance_domain():
    records = MODULE.read_jsonl(MODULE.DEFAULT_INPUT)
    sample = [MODULE.convert_record(record) for record in MODULE.select_sample(records)]

    assert len(sample) == 10
    counts = {}
    for question in sample:
        domain = question.metadata["finfactorbench"]["original_domain"]
        counts[domain] = counts.get(domain, 0) + 1
    assert counts == {domain: 2 for domain in MODULE.DOMAIN_SAMPLE_PLAN}
