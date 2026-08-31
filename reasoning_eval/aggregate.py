"""Join blind reasoning verdicts against withheld correctness and report.

The judges never saw ground truth, so crossing their scores with accuracy
separates traces that reasoned their way to the answer from ones that landed
on it anyway.
"""

import glob
import json
import os
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
LABEL = {
    "direct_forecast": "Direct Forecast",
    "structured_reasoning": "Structured Reasoning",
    "factor_memory": "Factor Memory",
    "principle_memory": "Principle Memory",
    "case_memory": "Case Memory",
    "structure_memory": "Structure Memory",
    "hgf": "HGF",
}
ORDER = list(LABEL)
HIGH, LOW = 4.0, 2.5


def truth(qid, method):
    path = (
        f"{REPO}/runs/fix_hgf/cases/{qid}/procedural_topology_hgf_canonical.json"
        if method == "hgf"
        else f"{REPO}/runs/fix_baselines/cases/{qid}/{method}.json"
    )
    with open(path) as fh:
        row = json.load(fh)
    return row["metrics"]["accuracy"], row["metrics"]["brier"]


def main():
    recs, malformed = [], []
    for path in sorted(glob.glob(f"{HERE}/verdicts/q*.json")):
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as exc:
            malformed.append((os.path.basename(path), str(exc)[:60]))
            continue
        qid = doc["question_id"]
        rank = doc.get("within_question_ranking") or []
        for v in doc["verdicts"]:
            m = v["method"]
            sc = v["scores"]
            acc, brier = truth(qid, m)
            recs.append(
                {
                    "qid": qid,
                    "method": m,
                    "score": st.mean(sc[d] for d in DIMS),
                    "dims": sc,
                    "flags": v.get("flags", {}),
                    "accuracy": acc,
                    "brier": brier,
                    "accept": v.get("would_a_careful_analyst_accept_this"),
                    "rank": rank.index(v["trace_id"]) + 1 if v["trace_id"] in rank else None,
                    "quote": v.get("decisive_quote", ""),
                    "line": v.get("verdict_one_line", ""),
                }
            )
    if malformed:
        print("  파싱 실패:", malformed)
    if not recs:
        print("  판정 결과 없음")
        return

    qs = sorted({r["qid"] for r in recs})
    print(f"  질문 {len(qs)}개 · 판정 {len(recs)}건\n")

    print(f"  {'방법':<22}{'추론점수':>8}{'정합성':>8}{'근거성':>8}{'논리':>7}{'확률':>7}{'기제':>7}{'순위':>7}{'정확도':>8}")
    print("  " + "-" * 84)
    for m in ORDER:
        rs = [r for r in recs if r["method"] == m]
        if not rs:
            continue
        dims = {k: st.mean(r["dims"][k] for r in rs) for k in DIMS}
        rk = [r["rank"] for r in rs if r["rank"]]
        print(
            f"  {LABEL[m]:<22}{st.mean(r['score'] for r in rs):>8.2f}"
            f"{dims['prediction_alignment']:>8.2f}{dims['evidence_grounding']:>8.2f}"
            f"{dims['logical_validity']:>7.2f}{dims['probability_justification']:>7.2f}"
            f"{dims['mechanism_specificity']:>7.2f}"
            f"{(st.mean(rk) if rk else float('nan')):>7.2f}"
            f"{st.mean(r['accuracy'] for r in rs):>8.2f}"
        )

    print("\n  === 추론 점수 × 정답 교차표 (심사자는 정답을 못 봄) ===")
    print(f"  {'방법':<22}{'정답+추론높음':>14}{'정답+추론낮음':>16}{'오답+추론높음':>16}{'오답+추론낮음':>16}")
    print("  " + "-" * 84)
    for m in ORDER:
        rs = [r for r in recs if r["method"] == m]
        if not rs:
            continue
        ok_hi = sum(1 for r in rs if r["accuracy"] == 1 and r["score"] >= HIGH)
        ok_lo = sum(1 for r in rs if r["accuracy"] == 1 and r["score"] <= LOW)
        no_hi = sum(1 for r in rs if r["accuracy"] == 0 and r["score"] >= HIGH)
        no_lo = sum(1 for r in rs if r["accuracy"] == 0 and r["score"] <= LOW)
        print(f"  {LABEL[m]:<22}{ok_hi:>14}{ok_lo:>16}{no_hi:>16}{no_lo:>16}")

    print("\n  === 플래그 발생률 (%) ===")
    print(f"  {'방법':<22}" + "".join(f"{f.split('_')[0][:9]:>11}" for f in FLAGS))
    print("  " + "-" * 84)
    for m in ORDER:
        rs = [r for r in recs if r["method"] == m]
        if not rs:
            continue
        cells = "".join(
            f"{100*sum(1 for r in rs if r['flags'].get(f)) / len(rs):>11.0f}" for f in FLAGS
        )
        print(f"  {LABEL[m]:<22}{cells}")

    lucky = sorted(
        (r for r in recs if r["accuracy"] == 1 and r["score"] <= LOW),
        key=lambda r: r["score"],
    )
    print(f"\n  === 찍어서 맞춘 것으로 보이는 사례 {len(lucky)}건 ===")
    for r in lucky[:10]:
        print(f"  [{r['score']:.1f}] {LABEL[r['method']]:<22}{r['qid'][:44]}")
        print(f"        {r['line'][:150]}")

    sound = sorted(
        (r for r in recs if r["accuracy"] == 0 and r["score"] >= HIGH),
        key=lambda r: -r["score"],
    )
    print(f"\n  === 논리는 타당했으나 빗나간 사례 {len(sound)}건 ===")
    for r in sound[:6]:
        print(f"  [{r['score']:.1f}] {LABEL[r['method']]:<22}{r['qid'][:44]}")

    paper_table(recs, len(qs))
    with open(f"{HERE}/joined.json", "w") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2)
    print(f"\n  → {HERE}/joined.json")


def seed_metrics():
    """Three-seed Brier/accuracy/NLL, so scores sit beside the headline numbers."""
    runs = {
        0: ("runs/fix_baselines", "runs/fix_hgf"),
        1: ("runs/seed1_baselines", "runs/seed1_hgf"),
        2: ("runs/seed2_baselines", "runs/seed2_hgf"),
    }
    out = {m: {"accuracy": [], "brier": [], "nll": []} for m in ORDER}
    for base, hgf in runs.values():
        with open(f"{REPO}/{base}/results.json") as fh:
            b = json.load(fh)["summary"]["overall"]
        with open(f"{REPO}/{hgf}/results.json") as fh:
            h = json.load(fh)["summary"]["overall"]
        for m in ORDER:
            src = h if m == "hgf" else b.get(m)
            if not src:
                continue
            for k in ("accuracy", "brier", "nll"):
                out[m][k].append(src[k])
    return out


def paper_table(recs, n_questions):
    seeds = seed_metrics()
    print("\n\n  ══ 논문용 표 ══")
    print(f"  추론 점수는 시드 0 트레이스 {len(recs)}건에 대한 LLM 심사 (질문 {n_questions}개 × 7방법),")
    print("  Brier/Acc/NLL 은 시드 0·1·2 평균±SD. 심사자는 정답을 보지 못함.\n")
    print(
        f"  {'Method':<24}{'Acc':>14}{'Brier':>16}{'NLL':>16}"
        f"{'추론점수':>10}{'운으로맞춘비율':>14}"
    )
    print("  " + "-" * 96)
    for m in ORDER:
        rs = [r for r in recs if r["method"] == m]
        s = seeds[m]
        if not rs or not s["brier"]:
            continue
        correct = [r for r in rs if r["accuracy"] == 1]
        lucky = sum(1 for r in correct if r["score"] <= LOW)
        rate = f"{100 * lucky / len(correct):.0f}% ({lucky}/{len(correct)})" if correct else "-"
        sd = {k: (st.stdev(s[k]) if len(s[k]) > 1 else 0.0) for k in ("accuracy", "brier", "nll")}
        print(
            f"  {LABEL[m]:<24}"
            f"{st.mean(s['accuracy']):>8.3f}±{sd['accuracy']:<5.3f}"
            f"{st.mean(s['brier']):>9.4f}±{sd['brier']:<6.4f}"
            f"{st.mean(s['nll']):>9.4f}±{sd['nll']:<6.4f}"
            f"{st.mean(r['score'] for r in rs):>10.2f}"
            f"{rate:>14}"
        )
    print(
        "\n  '운으로맞춘비율' = 정답을 맞힌 트레이스 중 추론 점수가 "
        f"{LOW} 이하인 비율. 낮을수록 정확도가 추론에 뒷받침됨."
    )


if __name__ == "__main__":
    main()
