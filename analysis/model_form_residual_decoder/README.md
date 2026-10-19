# Pre-UDE gate result

This experiment asks whether a small neural discrepancy can repair an
incorrect SAC feedback form. The network predicts the saturating-SAC minus
prior discrepancy from time, imposed length, the prior V/Ca/T trace, and the
baseline trace. It is trained on five protocols and 35 cells; a long-hold
protocol and 12 cells are sealed.

The result is a **negative gate**:

| Output | Prior model | Residual decoder |
|---|---:|---:|
| V MAE | 0.0203 mV | 0.1967 mV |
| Ca MAE | 0.000075 µM | 0.0235 µM |
| T MAE | 0.000466 kPa | 0.1336 kPa |
| Overall mixed-unit MAE | 0.00695 | 0.1179 |

The free output residual did not generalize to the sealed feedback-form shift;
it made the already-good prior worse. This does not disprove UDEs in general,
and it is not a full flux-level UDE. It does show that adding a neural
correction without first separating physiological-state error from feedback
form error is unsafe. A future UDE should therefore be restricted to an
explicit SAC/TnC flux, include a mechanism gate, and be tested against this
negative output-residual baseline.

Files:

- `heldout_residual_comparison.png` / `.pdf`: truth, prior, and correction
- `metrics.json`: sealed test metrics
- `model_form_residual_decoder.py`: reproduction script
