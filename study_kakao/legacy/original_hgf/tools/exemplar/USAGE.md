# Worked-exemplar generator

This directory preserves the v22 worked-exemplar generation path that produced
the fixed artifacts under `artifacts/exemplars`.

The generator:

1. loads one resolved memory question and its validated WorldReasoner DAG;
2. removes outcome and post-cutoff information;
3. passes the transferable DAG structure and cutoff-safe articles to the model;
4. validates article citations and required reasoning fields;
5. performs one repair call when validation fails; and
6. saves both the raw cache entry and a `fixed_memory_exemplar_v1` artifact.

The prompt, schema, seed role, validation, and repair behavior come from
`icaif/forecaster/run_dag_exemplar_subset.py::_distill_exemplar`.

## Usage

From `study_kakao`:

```bash
python exemplar/generate.py \
  v3_aapl_revenue_growth_acceleration_2023_04_01
```

Set `OPENROUTER_API_KEY` in the environment or `study_kakao/.env` first. The
historical defaults are retained:

- model: `google/gemini-2.5-flash-lite`
- maximum output tokens: `2400`
- temperature: `0`
- seed role: `dag-exemplar`

Generated wrappers go to `exemplar/generated/` and raw cache entries go to
`exemplar/cache/`. These paths are separate from the frozen reproduction
artifacts, so rerunning the generator does not overwrite them.

Multiple memory-question IDs may be supplied in one command. Use `--model`,
`--max-tokens`, `--output-dir`, and `--cache-dir` to override the defaults.
