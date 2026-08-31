# Provenance

This directory preserves the complete artifact-generation source used for the
reported ReFoRM experiments. It is isolated from the portable forecasting
package so its Blueprint and exemplar dependencies cannot be mixed with runtime
packaging changes.

`scripts/build_hgf_artifacts.py` executes this snapshot with the fresh artifact
root supplied through `HGF_ROOT`.
