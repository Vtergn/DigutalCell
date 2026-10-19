# Exploratory feedback-gain ablation after the frozen 64-cell regime gate

Date: 2026-09-29. This experiment was designed **after** observing the 64-cell high-load regime map and is therefore a follow-up mechanism probe, not independent validation of regime prevalence or predictive performance.

For each of the three pre-labeled period-2 candidate backgrounds (`train_016`, `train_036`, `test_006`), hold basal ICaL and SERCA scales fixed at the values in `reverse_robust3/design.csv`. At `CL=0.10`, 1000-ms pacing and common initial state, run 100 beats at feedback gains **0** and **0.7**. Reuse the pre-existing native-gain 100-beat outputs as the third point. Apply exactly the same frozen beat labels from `mct_regime100_protocol.md` without changing criteria.

Question: does reducing the implemented MCT feedback gain remove the late alternating APD/Ca pattern in these specific virtual parameter backgrounds? This is a model ablation, not evidence of clinical arrhythmia or of a general monotonic bifurcation curve.
