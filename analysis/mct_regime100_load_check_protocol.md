# Exploratory load-dependence check for three high-load period-2 candidates

Date: 2026-09-29. Written after the frozen 64-cell high-load gate and the feedback-gain ablation. This is a selected-background mechanism check, not an independent prevalence estimate.

For each of `train_016`, `train_036`, `test_006`, keep the native MCT gain, basal ICaL scale and SERCA scale from `reverse_robust3/design.csv`. From the common initial state, simulate 100 beats at 1000-ms pacing under `CL=0.02` and `CL=0.06`; reuse their existing `CL=0.10` traces. Apply the unchanged frozen beat labels in `mct_regime100_protocol.md`.

Question: is the late period-2 candidate pattern observed only under higher mechanical load within each of these parameter backgrounds? A low/mid stable and high alternating result supports load-conditioned model dynamics, but does not identify an exact critical load, prove a bifurcation, or validate biological arrhythmia.
