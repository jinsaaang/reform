# Dependency Audit

## Scope

The audit covers the code under `hgf_final` and the public launchers under
`scripts`. The source was extracted from canonical archive commit
`dbf76bce37955e7ef5f287e70b7dc56556446971`.

## Runtime dependencies

The only third-party Python packages imported by the final HGF runtime are
`openai`, `pydantic`, and `python-dotenv`. SQLite support comes from the Python
standard library. Tests additionally require `pytest`.

## Internal dependencies

The final HGF method imports only modules shipped in this branch.

- `hgf_e2e_topology` contains the forecasting procedure.
- `hgf` contains frozen shared question, evidence, retrieval, generation,
  validation, and boundary utilities.
- `hgf_original_input_adapter` supplies optional frozen evidence and retrieval.
- `hgf_e2e_topology_sidecar` and
  `hgf_e2e_topology_provider_pinned` record calls and enforce provider routing.

There are no imports from exploratory HGF packages, no reads from previous run
directories, and no absolute `/home` or `/tmp` path in executable source.

The name `hgf_historical_base_src` means a frozen internal utility layer. It is
part of this branch and is not checked out or loaded from another commit at
runtime. The method protocol records `previous_experiment_packages: []` and
`previous_result_files: []`.

## Data dependencies

Executable code is self-contained. Questions, evidence databases, Blueprints,
and answer-free exemplars are intentionally external benchmark inputs. This is
a data contract, not a code-version dependency. Frozen replay additionally
requires the two model-specific input manifests supplied on the command line.

## Baseline caveat

The shared baseline module is the frozen controlled-comparison implementation.
It contains dead code for an earlier HGF arm, but the public launcher restricts
selection to six baselines and cannot call it. Direct DAG and resolved-case
inputs require the same semantic leakage audit used in the paper experiment
before their numbers are reportable. The portable launcher does not silently
rewrite user-provided DAG or case artifacts.

