# Reasoning Judge Rubric

You are grading **forecast reasoning traces**. Seven methods each answered the
same question from the same evidence bank. Your job is to judge how well each
one reasoned — **not** whether it turned out to be right.

## Hard constraints

1. **The correct answer is deliberately withheld.** Do not try to determine it,
   look it up, or infer it from outside knowledge. Do not read any file other
   than your assigned packet and this rubric. Do not search the repository, the
   `runs/` directory, the `data/` directory, or the web. Doing so invalidates
   the study.
2. **Judge each trace against the inputs it actually received.** By deliberate
   experimental design the methods do **not** share one evidence bank:
   `direct_forecast`, `structured_reasoning`, `case_memory`, `principle_memory`
   and `structure_memory` read bank **E0**; `factor_memory` and the HGF method
   read bank **E1**. Both banks are in the packet under `evidence_banks`, and
   every trace names its own in `evidence_bank_used`. Memory-based methods
   additionally received a `retrieved_memory` block carried over from earlier
   questions. Check a trace's grounding against **its own bank plus its
   retrieved memory only**. An id that is absent from E0 but present in E1 is
   correctly cited by an E1 method — that is not a hallucination. Never
   penalise a trace for failing to use a bank it was never given, and never
   reward or penalise a method for the size of its bank.
3. **Do not reward verbosity, jargon, or structural formatting.** A long trace
   with an unsupported leap scores lower than a short trace that states its
   limits and matches its confidence to them. Method names are visible; ignore
   them and grade the content.
4. **Every score needs a quote.** Cite the exact span that drove it.

## Scores — five dimensions, integers 1–5

### 1. `evidence_grounding`
Are factual and numeric claims traceable to the cited evidence?

- **1** — Invents figures, or misquotes evidence in a way that changes the conclusion.
- **3** — Mostly grounded, but at least one material claim is asserted without support.
- **5** — Every factual and numeric claim traces to evidence in the bank; citations match what the source actually says.

### 2. `logical_validity`
Do the premises connect to the conclusion?

- **1** — Non-sequitur, or contradicts itself.
- **3** — Broadly follows, but a step is skipped or a competing consideration is dropped without comment.
- **5** — Each step follows from the prior; competing considerations are resolved explicitly rather than ignored.

### 3. `prediction_alignment`
Does the stated reasoning actually entail the option that was chosen? **This is
the central test.** Read the reasoning, decide what it implies, then compare to
`forecast.prediction`.

- **1** — The reasoning points to a different option than the one selected.
- **3** — The reasoning is consistent with the selection but equally consistent with another option; the choice is underdetermined.
- **5** — The reasoning uniquely and explicitly selects the chosen option, including why neighbouring options are excluded.

### 4. `probability_justification`
Does the probability mass match the uncertainty the trace itself declares?

- **1** — Declares a claim unsupported or unverifiable, then assigns high confidence anyway (or the reverse: strong evidence, near-uniform mass).
- **3** — Direction of confidence is defensible but the magnitude is asserted, not reasoned.
- **5** — The distribution is argued for: stated support level, admitted gaps, and mass allocation line up.

### 5. `mechanism_specificity`
Is there a concrete causal mechanism, or generic filler?

- **1** — Boilerplate that would fit any question on any indicator.
- **3** — Names real drivers, but does not connect them to the target quantity.
- **5** — Traces a specific mechanism from named drivers through to the target quantity and its boundaries.

## Binary flags — `true` / `false`, each needs a quote when true

- `unsupported_magnitude_leap` — moves from directional evidence to a precise magnitude or boundary claim with no supporting quantity.
- `hallucinated_number` — states a figure that appears nowhere in the evidence bank.
- `internal_contradiction` — asserts something and then relies on its negation.
- `boilerplate_only` — no question-specific content beyond restating the prompt.
- `post_hoc_option_fit` — reads as if the option was chosen first and the rationale assembled afterwards.
- `admits_own_gap` — explicitly names the step it could not verify. **This is a
  positive signal**, not a defect; record it so honesty can be separated from
  hedging.

## Output

Write **strict JSON only** to the output path you are given — no prose, no
markdown fence. Schema:

```json
{
  "question_id": "<from packet>",
  "verdicts": [
    {
      "trace_id": "<from packet>",
      "method": "<from packet>",
      "scores": {
        "evidence_grounding": 1,
        "logical_validity": 1,
        "prediction_alignment": 1,
        "probability_justification": 1,
        "mechanism_specificity": 1
      },
      "flags": {
        "unsupported_magnitude_leap": false,
        "hallucinated_number": false,
        "internal_contradiction": false,
        "boilerplate_only": false,
        "post_hoc_option_fit": false,
        "admits_own_gap": false
      },
      "decisive_quote": "<exact span from the trace that drove the scores>",
      "verdict_one_line": "<one sentence>",
      "would_a_careful_analyst_accept_this": "yes | with_reservations | no"
    }
  ],
  "within_question_ranking": ["<trace_id best>", "...", "<trace_id worst>"],
  "ranking_rationale": "<two sentences on what separated top from bottom>",
  "shared_evidence_note": "<one sentence: did the methods disagree because they read different evidence, or because they reasoned differently from the same evidence?>"
}
```

All seven traces must appear in `verdicts` and in `within_question_ranking`.
