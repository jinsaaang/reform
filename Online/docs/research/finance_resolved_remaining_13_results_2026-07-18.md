# Finance Resolved Backtest: Remaining 13 Questions

## Executive summary

The remaining resolved-question backtest completed successfully for all 13 questions and all three forecasting arms.

The result does **not** support the current sparse-memory DAG treatment as an accuracy-improving forecasting method. Search-only slightly improved mean Brier score over direct forecasting, while adding retrieved DAG memory worsened both mean Brier score and majority accuracy. This is not a general negative result for well-covered finance DAG retrieval: the cutoff policy left only one to three eligible historical episodes, and nearly every target received the same unrelated memories.

At the same time, the LLM judge usually preferred the reasoning produced with DAG memory over direct reasoning. This preference did not align with realized outcomes: for Direct vs Search+DAG, the judge favored Search+DAG in 7 questions and Direct in 2, with 4 no-quorum ties, while Search+DAG had a better Brier score in only 3 questions and a worse score in 10.

The useful conclusion is therefore diagnostic rather than confirmatory:

- date-filtered search is promising but the effect is small;
- the current DAG retrieval/injection can make a reasoning path look more persuasive without improving its probability estimate;
- the present LLM-judge protocol is not a reliable proxy for forecast correctness and must be reported separately from outcome metrics;
- the next experiment should improve DAG relevance and probability integration rather than simply supply more DAG context.

## Experiment setup

| Item | Value |
|---|---|
| Cohort | 13 resolved binary finance questions left after excluding the prior 10-question experiment; run as a 1-question validation pilot plus a 12-question suite |
| Repetitions | 1 per question |
| Arm A | Direct forecasting |
| Arm B | Date-filtered search evidence + forecasting |
| Arm C | Date-filtered search evidence + retrieved historical DAG memory + forecasting |
| Forecast model | `openrouter/openai/o4-mini`, reasoning effort `high` |
| Judge panel | 3 instances of the same model; each pair evaluated in both presentation orders |
| Search source | WorldReasoner public-DB metadata proxy, filtered by each question's forecast cutoff |
| DAG retrieval | Lexical retriever, top 3 eligible historical episodes |
| Forecast calls | 39 |
| Judge calls | 234 |
| Successful paid model calls | 273/273 |

Each judge member evaluated A-B, B-C, and A-C twice with reversed candidate order. A member vote was usable only when its two orderings were consistent. This protects against simple position bias but also produced several no-quorum pair results.

## Outcome metrics

Lower Brier score is better.

| Arm | Mean Brier | Accuracy | Correct |
|---|---:|---:|---:|
| Direct | 0.2203 | 61.5% | 8/13 |
| Search-only | **0.2110** | 61.5% | 8/13 |
| Search+DAG | 0.2609 | 53.8% | 7/13 |

Relative comparisons:

- Search-only vs Direct: mean Brier improved by 0.0093; Search-only was better on 8 questions and worse on 5.
- Search+DAG vs Search-only: mean Brier worsened by 0.0499; Search+DAG was better on 3, worse on 9, and unchanged on 1.
- Search+DAG vs Direct: mean Brier worsened by 0.0406; Search+DAG was better on 3 and worse on 10.
- Direct to Search+DAG caused five majority-label flips. Two moved to the correct label and three moved to the wrong label, producing the net accuracy loss of one question.

These results make Search-only the best of the three tested arms for this cohort, but the sample is too small and has only one stochastic repetition per question. The 0.0093 Brier improvement should be treated as an encouraging pilot signal, not a stable effect estimate.

## Historical DAG coverage audit

The seed database contains 37 graph-built finance questions, but most resolve after the simulated cutoffs used by this cohort. Temporal eligibility therefore reduced the usable memory pool to the following:

| Target cutoffs | Target count | Eligible DAGs supplied | Historical sources |
|---|---:|---:|---|
| 2024-06-12 | 1 | 1 | Microsoft–Activision acquisition completion |
| 2024-12-31 through 2025-09-01 | 10 | 2 | Microsoft–Activision acquisition; Saudi petrodollar media-attention timing |
| 2025-10-08 through 2025-11-10 | 2 | 3 | The two above plus Dogecoin ETF approval |

This means that questions about CPI, EUR/USD, Chinese GDP, an S&P 500 correction, US debt, pharmaceutical pricing, and an IPO were mostly conditioned on the same Microsoft-acquisition and petrodollar DAGs. The current top-k retriever had no minimum relevance gate and no `no suitable DAG` fallback, so sparse coverage became forced irrelevant memory.

Accordingly, this pilot establishes that forced low-relevance DAG injection can hurt calibration. It does not yet test the intended hypothesis that sufficiently similar, mechanism-aligned finance DAGs improve forecasting.

## Reasoning-quality judge results

| Pair | First preferred | Second preferred | No-quorum tie | Outcome-based direction for second |
|---|---:|---:|---:|---|
| Direct vs Search-only | Direct 1 | Search-only 9 | 3 | Better 8, worse 5 |
| Search-only vs Search+DAG | Search-only 6 | Search+DAG 4 | 3 | Better 3, worse 9, unchanged 1 |
| Direct vs Search+DAG | Direct 2 | Search+DAG 7 | 4 | Better 3, worse 10 |

For Direct vs Search+DAG, only 9 questions yielded a directional judge preference. The judge preference agreed with the Brier direction on 4 and disagreed on 5. Thus, even after excluding no-quorum ties, the judge was below 50% concordance with realized probability quality in this small cohort.

Order consistency is also a concern:

- all 234 judge calls returned parse-valid responses;
- 35 of 117 judge-member pair evaluations were inconsistent after candidate order reversal;
- Direct vs Search+DAG had 14 inconsistent member evaluations out of 39;
- every panel-level tie was caused by failure to reach quorum, not by explicit tie votes.

The judge evidence therefore says that DAG-conditioned reasoning often appears more coherent or persuasive to the same model family, but it does not establish that the reasoning is more decision-useful.

## Per-question results

`p(Yes)` is the predicted probability of the positive label. A-C direction compares Search+DAG against Direct using Brier score.

| # | Question | Truth | Cutoff | Direct p(Yes) | Search p(Yes) | DAG p(Yes) | Direct Brier | DAG Brier | A-C outcome direction | A-C judge |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | Saudi Arabia withdraws from the petrodollar agreement | No | 2024-06-12 | 14.0% | 20.0% | 18.0% | 0.0196 | 0.0324 | Worse | DAG |
| 2 | SEC approves a Dogecoin ETF | Yes | 2024-12-31 | 35.0% | 34.0% | 33.0% | 0.4225 | 0.4489 | Worse | No quorum |
| 3 | US CPI falls below 2.5% in 2025 | Yes | 2025-01-01 | 48.0% | 51.5% | 62.0% | 0.2704 | 0.1444 | Better | DAG |
| 4 | EUR/USD above 1.10 at end of 2025 | No | 2025-01-01 | 46.0% | 29.5% | 31.0% | 0.2116 | 0.0961 | Better | No quorum |
| 5 | A start-of-2025 top-10 US company leaves the top 10 | Yes | 2025-01-01 | 60.0% | 34.0% | 33.0% | 0.1600 | 0.4489 | Worse | DAG |
| 6 | China's 2025 GDP growth exceeds its 2024 growth | No | 2025-01-01 | 38.0% | 31.0% | 60.0% | 0.1444 | 0.3600 | Worse | DAG |
| 7 | US economy enters a recession in 2025 | No | 2025-01-08 | 27.0% | 25.0% | 38.0% | 0.0729 | 0.1444 | Worse | Direct |
| 8 | US and EU reach a pharmaceutical-pricing agreement | No | 2025-03-02 | 30.5% | 20.0% | 31.0% | 0.0930 | 0.0961 | Worse | No quorum |
| 9 | S&P 500 experiences a 10% correction in 2025 | Yes | 2025-03-02 | 44.0% | 40.0% | 60.8% | 0.3136 | 0.1541 | Better | DAG |
| 10 | US national debt surpasses $40T by end of 2025 | No | 2025-03-02 | 31.0% | 15.0% | 54.0% | 0.0961 | 0.2916 | Worse | Direct |
| 11 | Medline completes an IPO over $6B in 2025 | Yes | 2025-09-01 | 30.0% | 44.5% | 27.8% | 0.4900 | 0.5220 | Worse | DAG |
| 12 | DraftKings launches a prediction-market platform | Yes | 2025-10-08 | 36.0% | 31.0% | 31.0% | 0.4096 | 0.4761 | Worse | DAG |
| 13 | NVIDIA beats the specified quarterly EPS estimate | Yes | 2025-11-10 | 60.0% | 62.0% | 58.0% | 0.1600 | 0.1764 | Worse | No quorum |

## Cost profile

The successful 13-question run consumed 1,563,603 input tokens and 517,510 output tokens, with provider-reported cost of approximately **$3.12**.

| Component | Calls | Cost |
|---|---:|---:|
| Direct forecasts | 13 | $0.124 |
| Search-only forecasts | 13 | $0.152 |
| Search+DAG forecasts | 13 | $0.384 |
| A-B judges | 78 | $0.623 |
| B-C judges | 78 | $0.946 |
| A-C judges | 78 | $0.891 |
| **Total** | **273** | **$3.120** |

The judge panel accounts for about 79% of cost. The main cost driver is not forecasting itself but six judge calls per arm pair per question, especially comparisons containing the longer DAG reasoning context.

## Interpretation and next experiment

The present evidence supports keeping the separation between searching and forecasting, but it does not support freezing the current DAG treatment as the proposed method.

The most plausible failure mode is that lexical retrieval supplies structurally related but decision-irrelevant causal material. The forecaster can then produce a more elaborate and apparently reasonable narrative while overweighting retrieved scenarios or anchoring on probabilities from a different historical context. The China GDP, US debt, Medline, and top-10-company questions are clear examples where DAG conditioning moved the probability substantially away from the resolved answer.

The next minimum experiment should make three targeted changes:

1. Add a retrieval gate using question similarity, forecast horizon, market regime, and causal-role overlap; allow the system to return no DAG when no episode clears the gate.
2. Separate scenario generation from probability adjustment. Require the forecaster to state which retrieved causal links are applicable, which are rejected, and the bounded probability contribution of each accepted link.
3. Reduce judge cost and improve validity. Use one order-balanced judge pass for development, reserve the full panel for the final evaluation, and measure judge/outcome concordance explicitly.

After these changes, rerun the same resolved cohort with at least three repetitions and paired bootstrap confidence intervals. The success criterion should require both improved Brier score and non-degraded reasoning quality; judge preference alone must not count as evidence of forecasting utility.

## Validity limitations

- This is a 13-question pilot with one repetition per arm, so sampling and model variance are not estimated.
- Search used a date-filtered public-DB metadata proxy, not a genuinely frozen historical web snapshot. The run should not be labeled fully leakage-free.
- Date filtering controls retrieved records, but it cannot prove that the model itself lacks post-cutoff knowledge.
- The forecast and judge models are from the same model family, so stylistic self-preference may inflate reasoning-quality scores.
- This cohort and protocol differ from the earlier 10-question live/unresolved experiment and should not be pooled with it as one homogeneous estimate.

## Reproducible artifacts

A compact machine-readable summary is tracked at `docs/research/finance_resolved_remaining_13_summary_2026-07-18.json`. Raw suites are generated artifacts and remain ignored by Git; share them through a controlled release or team storage when a full reasoning audit is required.

- Pilot suite: `artifacts/finance/experiments/resolved-remaining-pilot-1-rerun-2026-07-18/`
- Pilot outcome analysis: `artifacts/finance/experiments/resolved-remaining-pilot-1-rerun-2026-07-18-analysis/`
- Remaining 12 suite: `artifacts/finance/experiments/resolved-remaining-12-2026-07-18/`
- Remaining 12 outcome analysis: `artifacts/finance/experiments/resolved-remaining-12-2026-07-18-analysis/`

Each artifact directory includes SHA-256 receipts. The successful suite hashes are:

- pilot: `734c8aa21c708b6d98ed21ef4041b35eec4d7087488e932abf35aa29c6359159`
- remaining 12: `8e2a7be91fc556e6e8b3fd401b8e9c605c178ede77a3d5034c86fe61c3878c77`
