# Frozen risk-aware C3 protocol

Created before C3 simulations and fresh-seed confirmation scoring.

## Scientific target

Reduce rare long-hold induced secondary-voltage-event prediction failures in
the saturating-SAC synthetic truth family. This is a *conditional correct-form*
physiology-identification test, not real-cell or unknown-form validation.

## Development and confirmation split

- Development: existing independent 64-cell seed used previously for model
  development; retain only cells passing baseline, C1, C2 and long-hold QC.
- Final confirmation: generate a fresh 64-cell seed after selecting all
  inverse settings and C3; never use old confirmation cells 8/18/43/57 for C3
  choice, threshold tuning, or feature selection.
- Cell is the unit of fitting, cross-validation, bootstrap and reporting.

## Calibration and candidate C3

Baseline, C1=(150 ms, lambda 1.05, hold 10 ms), C2=(600 ms, lambda 1.10,
hold 10 ms). Six C3 candidates: lambda 1.10, onset 150 or 600 ms, hold 20,
35 or 50 ms. Keep the same 1-ms linear ramp on and off. The blinded target
remains (300 ms, lambda 1.10, hold 100 ms). A C3 is never the target waveform.

## Inverse comparison

Use the same fixed PLS20 architecture and feature extraction across arms.
Compare (a) normalized output clipped to [0,1], as before, with (b) true
parameter transformed to logit(normalized parameter), inverse-logit after
prediction. The synthetic prior places training parameters strictly inside
bounds. Do not tune the logit epsilon or PLS dimension using the final cohort.
No predicted parameter or target feature from confirmation is used to fit an
inverse. Compare both transforms on cross-fitted development long-hold risk;
choose the transform with fewer native-ODE QC failures, then lower no-C3
CVaR90 on its passing cells, before choosing C3. Parameter
MAE and exact-boundary fraction are diagnostics, not the selection criterion.
Include the no-C3 baseline under the selected transform.

## Development selection

For each candidate, use the same five-fold cell-held-out CV, re-prepace each
estimated parameter vector in the native ODE, and predict held-out development
long-hold. Exclude a candidate if its QC failure rate exceeds the no-C3 arm.
Among eligible candidates, choose lowest CVaR90 of cellwise target-window V
MAE; tie-break by p90, then mean, then shorter hold and earlier onset. Record
median and event metrics but never select using confirmation outcomes.

Secondary electrical-event metrics: first upward crossing of -50 mV after
mechanical onset, and total milliseconds above -50 mV in the [300,602) ms
target window. If neither truth nor prediction crosses, timing error is 0; if
only one crosses, report an event presence mismatch separately and omit the
undefined timing error from finite timing MAE. This rule is fixed now.

## Final confirmation

Freeze one C3 and inverse method. On a new seed, train inverse on all eligible
development cells and predict each confirmation cell from B+C1+C2 versus
B+C1+C2+C3. Evaluate the same cells and same QC policy. Primary paired
endpoint: CVaR90 (worst ceil(0.1*n) V MAE values), with cell bootstrap CI for
the between-arm difference. Secondary: p90, mean, median, count >5 mV,
simulation failures, second-event presence, crossing time and D>-50.

No UDE or adaptive selection is trained in this experiment. A positive
development result alone cannot establish generalization.
