# Reasoning-quality judgement

Brier and accuracy say whether a forecast landed. They do not say whether the
reasoning that produced it held up. This directory scores the reasoning itself,
independently of the outcome, so a correct answer reached by an unsupported
guess can be told apart from one reached by an argument.

An LLM judge (Claude Opus 5, one judge per question) graded all seven methods on
the 100-question benchmark: 700 traces from the seed-0 run.

## What the judge could and could not see

Given: the question, its options and cutoff; the full evidence bank the trace
actually read; and the trace's `reasoning`, `forecast` and `probabilities`.

Withheld: **the ground truth and every scored metric**. This is what makes the
lucky-guess measurement possible — the judge cannot rationalise a score toward
an answer it does not know.

Evidence banks are not shared across methods. `direct_forecast`,
`structured_reasoning`, `case_memory`, `principle_memory` and
`structure_memory` read bank E0; `factor_memory` and HGF read E1. Each trace is
graded against its own bank plus the memory block it retrieved, never against a
bank it was never given. `build_packets.py` refuses to emit a packet whose trace
cites an id outside its own bank.

Method names were visible to the judge. Scores therefore carry a possible
familiarity bias toward the method under test; see Limitations.

## Rubric

Five dimensions, 1–5 each, plus six binary flags. Full anchors in `RUBRIC.md`.

| Dimension | Question it asks |
| --- | --- |
| `evidence_grounding` | Do the factual and numeric claims trace to the cited evidence? |
| `logical_validity` | Do the premises connect to the conclusion? |
| `prediction_alignment` | Does the stated reasoning actually entail the option chosen? |
| `probability_justification` | Does the probability mass match the uncertainty the trace declares? |
| `mechanism_specificity` | Is there a concrete causal mechanism, or generic filler? |

## Results

Reasoning score is the mean of the five dimensions over 100 questions.
Acc/Brier/NLL are the three-seed means from the forecasting runs.

| Method | Acc | Brier | NLL | Reasoning | Correct-but-unreasoned |
| --- | --- | --- | --- | --- | --- |
| Direct Forecast | 0.523±0.021 | 0.2349±0.0076 | 1.0405±0.0342 | 2.93 | 24% (13/54) |
| Structured Reasoning | 0.387±0.031 | 0.2852±0.0148 | 1.2099±0.0555 | 2.86 | 29% (12/42) |
| Factor Memory | 0.523±0.012 | 0.2351±0.0050 | 1.0007±0.0182 | 3.17 | 19% (10/53) |
| Principle Memory | 0.460±0.035 | 0.2595±0.0152 | 1.1164±0.0670 | 2.97 | 30% (13/44) |
| Case Memory | 0.453±0.042 | 0.2617±0.0076 | 1.1605±0.0494 | 2.96 | 24% (10/42) |
| Structure Memory | 0.483±0.012 | 0.2502±0.0050 | 1.0937±0.0026 | 2.86 | 37% (18/49) |
| **HGF** | **0.530** | **0.2123** | **0.9141** | **3.73** | **0% (0/53)** |

Crossing reasoning score against correctness separates the two ways of being
right. `Correct-but-unreasoned` counts traces that got the answer with a
reasoning score at or below 2.5.

| Method | Right, reasoned | Right, unreasoned | Wrong, reasoned | Wrong, unreasoned |
| --- | --- | --- | --- | --- |
| **HGF** | **19** | **0** | **20** | 1 |
| Factor Memory | 11 | 10 | 4 | 12 |
| Principle Memory | 6 | 13 | 1 | 11 |
| Structure Memory | 5 | 18 | 0 | 18 |
| Direct Forecast | 3 | 13 | 1 | 11 |
| Structured Reasoning | 1 | 12 | 1 | 16 |
| Case Memory | 1 | 10 | 0 | 15 |

HGF leads all five dimensions, ranks first in 52 of 100 questions against 17 for
the next method, and is the only method with no correct-but-unreasoned trace.
Its 0.56 lead over second place is about 6.5 times the standard error of that
difference. It also holds the largest wrong-but-reasoned count, which is the
expected shape for a method whose errors come from the evidence rather than from
the argument.

Flag rates, as a percentage of each method's 100 traces:

| Method | Unsupported magnitude leap | Hallucinated number | Internal contradiction | Boilerplate only | Post-hoc option fit | Admits own gap |
| --- | --- | --- | --- | --- | --- | --- |
| **HGF** | **29** | **0** | **23** | **0** | 12 | 100 |
| Direct Forecast | 47 | 3 | 46 | 11 | 12 | 100 |
| Case Memory | 49 | 2 | 38 | 11 | 15 | 96 |
| Principle Memory | 50 | 0 | 40 | 13 | 16 | 98 |
| Structure Memory | 59 | 4 | 47 | 12 | 22 | 95 |
| Factor Memory | 62 | 3 | 41 | 8 | 20 | 94 |
| Structured Reasoning | 64 | 0 | 23 | 0 | 23 | 45 |

## The dominant failure mode

Most correct-but-unreasoned traces share one shape: a validation failure sends
the method into a fallback that cites no evidence and declares an explicit
abstention with a 50/50 split, after which the forecast stage invents a central
estimate and commits 0.65 to 0.95 of the mass to it. On
`v3_fed_balance_sheet_monthly_growth_2025_12`, `direct_forecast` and
`principle_memory` emit content-identical fallbacks, which places the behaviour
in shared pipeline code rather than in either method.

## Limitations

- One judge per trace. Two questions were graded twice by accident and the
  low-scoring traces moved by up to 11 points on the 25-point scale, so
  individual trace scores are noisier than the per-method means suggest. A
  duplicate-judge subsample would quantify this properly.
- The judge saw method names. HGF also carries a distinctive output schema, so
  blinding was not achievable without discarding the content under test.
- Reasoning scores come from the seed-0 run only; Acc/Brier/NLL average three
  seeds. The columns are not from the same population and should not be
  correlated within a row.

## Reproducing

```bash
python eval/reasoning_judge/build_packets.py <question-ids.json>
python eval/reasoning_judge/validate.py
python eval/reasoning_judge/aggregate.py
```

`build_packets.py` reads the case rows under `runs/` and writes one
ground-truth-free packet per question into `packets/` (about 13 MB, not
committed). Judging itself is the manual step: each packet goes to one judge
along with `RUBRIC.md`, and the verdict lands in `verdicts/qNN.json`.
`validate.py` rejects malformed, truncated or incomplete verdicts;
`aggregate.py` joins them against the withheld metrics and prints the tables
above. `joined.json` holds all 700 scored traces with their decisive quotes.
