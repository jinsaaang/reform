"""Build ground-truth-blind judge packets, one per question.

Each packet carries the question, the full evidence bank, and every method's
reasoning trace. Ground truth and scored metrics are withheld so the judge
rates reasoning quality independently of whether the answer was right.
"""

import html
import json
import os
import re
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__)) + "/packets"

METHODS = [
    ("direct_forecast", "Direct Forecast"),
    ("structured_reasoning", "Structured Reasoning"),
    ("factor_memory", "Factor Memory"),
    ("principle_memory", "Principle Memory"),
    ("case_memory", "Case Memory"),
    ("structure_memory", "Structure Memory"),
    ("hgf", "Procedural Topology HGF"),
]

WITHHELD = {"ground_truth", "metrics", "references"}


def case_path(qid, method):
    if method == "hgf":
        return f"{REPO}/runs/fix_hgf/cases/{qid}/procedural_topology_hgf_canonical.json"
    return f"{REPO}/runs/fix_baselines/cases/{qid}/{method}.json"


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def evidence_for(db_path):
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "select id, title, content, published_date, source from articles order by published_date"
    ).fetchall()
    con.close()
    return [
        {
            "evidence_id": r[0],
            "title": clean(r[1]),
            "published": r[3],
            "source": r[4],
            "text": clean(r[2])[:1800],
        }
        for r in rows
    ]


def build(qid, index):
    traces = []
    question = options = cutoff = None
    bank_paths = {}
    for method, label in METHODS:
        with open(case_path(qid, method)) as fh:
            row = json.load(fh)
        question = row["question"]
        options = row["options"]
        cutoff = row.get("cutoff")
        tag, db = row.get("evidence_bank"), row.get("evidence_db")
        if not tag or not db:
            raise ValueError(f"{qid}/{method}: missing evidence bank tag or path")
        if bank_paths.setdefault(tag, db) != db:
            raise ValueError(
                f"{qid}: bank {tag} resolves to two paths — {bank_paths[tag]} and {db}"
            )
        memory_src = row.get("retrieved_memory_question_ids") or row.get(
            "retrieved_memory_question_id"
        )
        traces.append(
            {
                "trace_id": f"{qid}::{method}",
                "method": method,
                "method_label": label,
                "evidence_bank_used": tag,
                "retrieved_memory_from": memory_src,
                "retrieved_memory": row.get("memory"),
                "reasoning": row["reasoning"],
                "forecast": {
                    k: v for k, v in row["forecast"].items() if k not in WITHHELD
                },
                "probabilities": row["probabilities"],
            }
        )
    banks = {tag: evidence_for(bank_paths[tag]) for tag in sorted(bank_paths)}
    for trace in traces:
        known = {a["evidence_id"] for a in banks[trace["evidence_bank_used"]]}
        cited = set(trace["reasoning"].get("selected_evidence_ids") or [])
        if cited - known:
            raise ValueError(
                f"{trace['trace_id']}: cites {len(cited - known)} ids absent from its "
                f"own bank {trace['evidence_bank_used']} — packet would misrepresent it"
            )
    packet = {
        "question_id": qid,
        "question": question,
        "options": options,
        "information_cutoff": cutoff,
        "instruction_note": (
            "Ground truth and all scored metrics are deliberately withheld. "
            "Rate each trace on its own reasoning merits only. Methods did NOT "
            "all see the same evidence: each trace names its bank in "
            "'evidence_bank_used', and memory-based methods additionally saw the "
            "'retrieved_memory' block carried over from an earlier question. "
            "Check a trace's grounding against ITS OWN bank plus its retrieved "
            "memory — never against a bank it was not given."
        ),
        "evidence_banks": banks,
        "evidence_bank_sizes": {k: len(v) for k, v in banks.items()},
        "traces": traces,
    }
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/q{index:02d}_{qid}.json"
    with open(out, "w") as fh:
        json.dump(packet, fh, ensure_ascii=False, indent=2)
    return out, packet


def main():
    with open(sys.argv[1]) as fh:
        qids = json.load(fh)
    manifest = []
    for i, qid in enumerate(qids, 1):
        out, packet = build(qid, i)
        leak = [k for k in ("ground_truth", "metrics") if k in json.dumps(packet)]
        manifest.append(
            {
                "index": i,
                "question_id": qid,
                "path": out,
                "chars": os.path.getsize(out),
                "evidence": packet["evidence_bank_sizes"],
                "traces": len(packet["traces"]),
                "leak": leak,
            }
        )
    with open(f"{OUT}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    for m in manifest:
        flag = "LEAK " + ",".join(m["leak"]) if m["leak"] else "ok"
        print(
            f"  q{m['index']:02d}  {m['question_id'][:46]:<46}"
            f"evidence {m['evidence']!s:<22} traces {m['traces']}  "
            f"{m['chars']:>7,}자  {flag}"
        )


if __name__ == "__main__":
    main()
