# Predefined first gate for a mechanical-challenge dynamic-regime benchmark

Date: 2026-09-29. This protocol is frozen before the uniform long high-load simulations. It is a **rabbit MCT2024 simulator-domain exploration**, not a claim about experimental cardiomyocytes.

## Cohort and observation

Reuse the 48 training and 16 off-grid test parameter vectors from `analysis/mct_feedback_audit/reverse_robust3/design.csv`. Run **all 64** at `CL=0.10`, 1000-ms pacing for **100 beats** from the same author-supplied initial state. `train_036` was already run to 100 beats and its exact saved trace may be reused. No high-load test output enters an inverse or predictive fit. This first gate evaluates label viability before training any regime predictor; low/middle 100-beat observations will be generated only if a usable high-load target exists.

## Frozen beat-derived labels

Compute APD as milliseconds above −67 mV and Ca peak for beats 81–100. Label based on the **last 10 beats**:

1. `period2_candidate` if odd/even mean APD differs by at least 15 ms, odd/even mean Ca peak differs by at least 0.03 µM, the APD range **within each parity** is at most 10 ms, the Ca range within each parity is at most 0.02 µM, and successive APD differences change sign at least 8 of 8 possible interior turns. This is a pattern label, not a biological arrhythmia diagnosis.
2. Otherwise `stable_1to1` if last-ten APD range is at most 5 ms, last-ten relative Ca-peak range is at most 1%, and the means of beats 81–90 versus 91–100 differ by at most 5 ms APD and 0.02 µM Ca.
3. Otherwise `drifting_or_complex`. This class explicitly combines slow adaptation, irregular dynamics, and unclassified periodic behavior; it is not a mechanistic state.

Report last-ten APD/Ca median as a **descriptive magnitude**; for period-2 candidates also report parity-specific means. Do not treat a period-2 median as a physiological steady APD.

## Viability gate

Before building a classifier, require at least five `period2_candidate` training variants and at least two test variants, plus at least five `stable_1to1` in both splits. Otherwise, stop this cohort as a regime-discovery map and do not claim a supervised prediction benchmark. Inspect solver-setting sensitivity for any new period-2 candidate before mechanistic interpretation. No category label or threshold will be changed in response to observed prevalence in this run.

If the gate passes, the next stage will generate uniform long low/middle-load calibration traces, fit target-blind native-ODE point twins and direct nonlinear predictors, and evaluate both late magnitude and held-out high-load regime. If it fails, the next scientific question is whether the current parameter range/protocol actually populates multiple regimes; class enrichment must be a separately designed experiment.
