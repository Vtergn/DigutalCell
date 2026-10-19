# Flux-residual UDE gate

This diagnostic restricts the learned correction to a derivative residual and
forces it to zero at mechanical baseline. It uses five protocols and 35 cells
for training, then seals a long-hold protocol and 12 cells. The prior trace is
used to supply the known derivative; therefore this is not yet a native
ToR-ORd UDE implementation.

The diagnostic failed in its current form:

| Output | Prior | Flux residual |
|---|---:|---:|
| V MAE | 0.0203 mV | 47.53 mV |
| Ca MAE | 0.000075 µM | 0.1225 µM |
| T MAE | 0.000466 kPa | 2.964 kPa |
| Overall mixed-unit MAE | 0.00695 | 16.87 |

The learned derivative correction is amplified by integration drift. This is a
useful no-go result for an unconstrained derivative residual: a small local
flux error can produce a large long-time trajectory error. A real UDE would
need native state variables, bounded corrections, stable integration, and
mechanism-specific constraints before it should be considered.

Files:

- `comparison.png` / `.pdf`: truth, prior, and integrated correction
- `metrics.json`: sealed metrics
- `flux_residual_ude_gate.py`: reproduction script
