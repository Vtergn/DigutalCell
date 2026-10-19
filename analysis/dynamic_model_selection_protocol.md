# Locked 2×2 oracle decomposition and discrete SAC-model selection

This follows the C1/C2 physiological inverse result and precedes scoring the new target predictions. All 45 QC-complete confirmation virtual cells, the two existing 10-ms calibration probes, the same three sealed targets, waveform QC, and target windows remain unchanged.

## 2×2 decomposition on saturating-SAC truth

Use the existing baseline+C1/C2 inverse parameter vector and its newly re-paced 50-state initial condition as `theta_hat`. Compare **without a fitted SAC gain** to keep the axis clean:

1. `theta_hat` + original SAC: already generated native-ODE prior target.
2. true theta + original SAC: existing same-cell source-form waveform; oracle parameter diagnostic.
3. `theta_hat` + saturating SAC: new native-ODE prediction; correct form, estimated state.
4. true theta + saturating SAC: existing target truth (zero error by construction).

All predictions are baseline aligned for the primary error, using only the observed unperturbed waveform. Score long hold first, plus larger short and compound as checks. Report both mean and paired cell-bootstrap intervals on common QC-passing cells. The axes can interact, so do not attribute the joint error additively by default.

## Discrete mechanism selection

At a fixed `theta_hat`, simulate original, saturating and no-direct-SAC equations for C1/C2 and all three sealed targets. Use **only observed baseline/C1/C2**, together with the inferred-state candidate C1/C2 predictions, to select a mechanism. The prespecified hard rule is: direct-SAC gate <0.1 → no direct SAC; otherwise, C1/C2 effective-gain log-ratio > `0.05*shared_beta/2` → saturating SAC; otherwise original SAC. The gate, gain ratio, and shared beta were defined in the preceding flux pilot using the separate development cohort. No threshold is tuned on test-cell mechanism labels or target outcomes. This rule uses estimated-state compensation explicitly; an additional full-C1/C2 waveform-distance selector is diagnostic only.

Blind target prediction uses the selected native equation at the same inferred theta and state. Compare with source-form+continuous gate, correct-form oracle and strong simple baselines already scored. Report confusion matrices, all QC failures, V/Ca/tension waveform errors and per-target cell-bootstrap comparisons. The 2×2 oracle parameter arms are diagnostic only, not deployable twins.
