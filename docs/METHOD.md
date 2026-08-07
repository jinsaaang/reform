# Final Method and Baselines

## Procedural Topology HGF

The final method has six forecasting stages.

1. It reads the current cutoff-safe E1 evidence and writes an evidence ledger
   covering the target semantics, baseline, current drivers, counterevidence,
   and missing information.
2. It retrieves up to three outcome-redacted Blueprints from earlier resolved
   events with the same `family_id` and `target_metric`. Retrieval happens only
   after current evidence has been read.
3. It routes the Blueprint paths that are relevant to the current ledger. The
   original node order and edge relations are not rewritten.
4. It instantiates those paths with current evidence. Supported and
   contradicted relations can inform the forecast. Unverified historical
   relations are not treated as current facts.
5. It produces a flexible reasoning trace using verified partial topology,
   current factors missing from history, counterevidence, uncertainty, and an
   answer-free worked reasoning check. The exemplar supplies reasoning form,
   not a historical answer or probability.
6. A separate boundary call maps the model's direction and magnitude judgment
   to the public answer options and emits one probability distribution.

The Blueprint is therefore neither a prior forecast nor a complete chain that
must be followed. It is a reusable expert structure that helps the model check
and connect current evidence.

## Baselines

- `search_only` is structured direct forecasting. It uses current E0 evidence
  and the shared target and probability boundary, but no historical memory.
- `prospective_dag` constructs a DAG from the current E0 evidence and forecasts
  from that graph. It tests whether hindsight-derived structure is necessary.
- `direct_dag` retrieves a past DAG and gives it directly to the forecaster. It
  tests whether HGF's current instantiation and procedural use are necessary.
- `factor_memory` retrieves compact historical factors and uses E1 evidence. It
  is a strong non-topological memory baseline.
- `case_memory` retrieves resolved historical episodes as analogical memory.
- `text_memory` distills general forecasting principles from resolved events
  and removes the graph representation before forecasting.

HGF is not one of them. It is a separate implementation under
`hgf/method_src/hgf_e2e_topology` and is always run through
`scripts/run_hgf.py`.

## Evidence contract

The registered experiment used E0 for `search_only`, `prospective_dag`,
`direct_dag`, `case_memory`, and `text_memory`. It used E1 for `factor_memory`
and HGF. E0 and E1 are frozen evidence databases, not model predictions.
Paper comparisons should preserve this contract or explicitly report a new,
uniform evidence design as a separate experiment.

## Non-negotiable controls

- Every article must precede the forecast cutoff.
- Historical events must resolve before the current forecast cutoff.
- Retrieval requires an exact event family and target metric match.
- Historical outcomes and probabilities must not enter HGF.
- No method may read another method's prediction.
- No probability pooling, posterior adjustment, or result-conditioned retry is
  permitted.
- A successful run must retain current evidence IDs, retrieved Blueprint IDs,
  reasoning, raw requests and responses, token usage, cost, and latency.

