# v1.7.0 strict change record

Parent: `canonical_v1_6_0_strict`, copied from
`procedural_topology_hgf_canonical_v1_6_0_strict_20260803`.

## Forecast-method change

The v1.6.0 validator simultaneously required the central estimate's exact
public-interval mapping and prohibited an outer-range modal option when
magnitude support was weak. When zero lay outside the within-range interval,
this made a neutral zero estimate impossible to validate without inventing a
different central value.

v1.7.0 removes that prohibition. The exact arithmetic mapping remains strict;
weak support instead caps maximum option confidence at `0.50` for
`direction_only` and `0.45` for `insufficient`. The boundary prompt and
validator error message communicate the same joint rule.

The validator also emits one explicit joint repair instruction when the
arithmetic option is not modal: preserve the central estimate, mapped option,
and prediction, and change the probabilities so that arithmetic option is tied
for or above every alternative. This prevents prediction/argmax repair
oscillation without changing a model output in code.

## Execution change

`run_qwen_seed0.py` provides a persistent, resumable, two-trial maximum runner.
It uses fresh trial roots, merges only audit-reportable cases, rebuilds the
100-case aggregate from cache, records source hashes, and never uses `/tmp`.
