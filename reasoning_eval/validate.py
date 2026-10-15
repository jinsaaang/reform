"""Check every verdict file before it reaches the aggregate.

A retried judge overwrites its own verdict path, so only the last write
survives — but a write cut off midway leaves a file that still parses far
enough to fool a shallow check. Validate structure, coverage and ranges.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = [
    "evidence_grounding",
    "logical_validity",
    "prediction_alignment",
    "probability_justification",
    "mechanism_specificity",
]
FLAGS = [
    "unsupported_magnitude_leap",
    "hallucinated_number",
    "internal_contradiction",
    "boilerplate_only",
    "post_hoc_option_fit",
    "admits_own_gap",
]


def check(path):
    problems = []
    name = os.path.basename(path)
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        return [f"{name}: JSON 파손 — {exc}"]

    qid = doc.get("question_id")
    packet = f"{HERE}/packets/{name[:-5]}_{qid}.json"
    expected = None
    if os.path.exists(packet):
        with open(packet) as fh:
            expected = {t["trace_id"] for t in json.load(fh)["traces"]}

    seen = set()
    for v in doc.get("verdicts", []):
        tid = v.get("trace_id", "?")
        seen.add(tid)
        scores = v.get("scores") or {}
        missing = [d for d in DIMS if d not in scores]
        if missing:
            problems.append(f"{name}/{tid}: 점수 항목 누락 {missing}")
        bad = {k: s for k, s in scores.items() if not isinstance(s, int) or not 1 <= s <= 5}
        if bad:
            problems.append(f"{name}/{tid}: 점수 범위 이탈 {bad}")
        flags = v.get("flags") or {}
        if [f for f in FLAGS if f not in flags]:
            problems.append(f"{name}/{tid}: 플래그 누락")
        if not v.get("decisive_quote"):
            problems.append(f"{name}/{tid}: 인용구 없음")

    if len(seen) != 7:
        problems.append(f"{name}: 트레이스 {len(seen)}/7")
    if expected and seen != expected:
        problems.append(f"{name}: 트레이스 불일치 (누락 {sorted(expected - seen)})")

    ranking = doc.get("within_question_ranking") or []
    if set(ranking) != seen:
        problems.append(f"{name}: 순위 목록이 판정과 불일치 ({len(ranking)}개)")
    return problems


def main():
    paths = sorted(glob.glob(f"{HERE}/verdicts/q*.json"))
    all_problems = []
    for p in paths:
        all_problems += check(p)
    print(f"  판정 파일 {len(paths)}개 검사")
    if not all_problems:
        print("  이상 없음 ✅")
        return 0
    for line in all_problems:
        print(f"  ⚠ {line}")
    print(f"\n  문제 {len(all_problems)}건 — 해당 질문은 재심사 필요")
    return 1


if __name__ == "__main__":
    sys.exit(main())
