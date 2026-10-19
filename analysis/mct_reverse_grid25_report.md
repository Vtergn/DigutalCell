# Mechanical-challenge observability in a closed-loop rabbit virtual-cell grid

Date: 2026-09-29. **Scope:** exploratory, deterministic, same-model rabbit MCT2024 benchmark. This is a computational feasibility result for the reverse branch, not validation against real cells and not a human closed-loop cardiomyocyte.

## Design

We extended the previous two-load pilot to 25 fixed parameter variants: five feedback gains (`0.25–1.25`) crossed with five basal L-type Ca permeability scales (`0.90–1.10`). For each identical parameter vector, the native Shannon–Bers/Negroni/MCT2024 ODE was run under gel settings `CL=0.02, 0.06, 0.10`, with common supplied initial state, 1000-ms pacing, and 20 beats. The blind target was the final-beat **APD above −67 mV** and cytosolic Ca peak at `CL=0.10`. APD here is a threshold-duration measure, not APD90. `CL` is the model's gel crosslinker setting, not a measured force waveform.

Evaluation left out **one entire parameter variant** at a time. Its low and middle load observations were available for personalization; its high load response was never used to fit its inverse parameters or direct predictors. The middle-load observation was withheld in the low-only comparisons. The original benchmark rules were recorded in [`mct_reverse_grid25_protocol.md`](mct_reverse_grid25_protocol.md). The direct nonlinear RBF baseline was added **after** the predefined ridge/native-ODE scores as a stronger exploratory comparison.

## Main result

| Predictor | Calibration | High-load APD MAE (ms) | High-load Ca-peak MAE (µM) |
|---|---|---:|---:|
| No-change | low | 59.36 | 0.1650 |
| Direct ridge | low | 23.69 | 0.0662 |
| Native-ODE twin | low | 32.88 | 0.0935 |
| Direct ridge | low + mid | 20.14 | 0.0556 |
| Per-variant linear load extrapolation | low + mid | 24.16 | 0.0805 |
| Direct nonlinear RBF, post-score baseline | low + mid | 8.01 | 0.0236 |
| **Native-ODE twin** | **low + mid** | **1.28** | **0.00393** |

The mechanistic individualization step fit only a two-parameter **calibration-output forward RBF** on the other 24 variants; it did **not** use their high-load outputs to infer the held-out parameters. It then optimized the held-out variant's two parameters against its low or low+middle APD/Ca observations. Finally, it ran the **actual native MCT2024 ODE** at high load for the inferred parameters. This is a posterior-free point twin in a deliberately narrow, known-model setting.

Low-only calibration was insufficient: its MCT gain MAE was **0.427** over a total gain range of 1.0, and the resulting high-load twin was worse than direct ridge. Adding the middle mechanical challenge reduced gain MAE to **0.0187** and basal Ca-permeability-scale MAE from **0.0337** to **0.00123**. The low+mid native-ODE twin improved on the post-score direct nonlinear RBF in 23/25 variants for APD and 24/25 for Ca peak. Direct ridge itself improved with the middle load in 18/25 variants for each output, but by smaller amounts.

## Why this is biologically interpretable within the model

The low-load data allow compensation between basal L-type Ca permeability and mechanical-feedback gain. For five variants deliberately chosen for their largest low-only gain errors, the incorrectly inferred parameter sets were re-run through the **original ODE at the observed low load**: their APD calibration error was 0 ms and Ca-peak error averaged only 0.00024 µM. Thus the low-only parameter ambiguity in those examples is not merely an emulator fitting artifact. In a separate five-variant grid-spread audit of low+mid inversions, native-ODE calibration errors averaged 0.8 ms and 0.00228 µM across the observed conditions.

Middle load changes the feedback exposure. It makes the two parameter effects more distinguishable, so the inferred gain can be carried to a stronger load. This is the model-internal version of the central hypothesis: **a controlled mechanical challenge can add observability of a hidden electrical/Ca feedback state**. The model's feedback-null ablation from the preceding pilot independently showed that load-dependent AP/Ca changes vanish when the implemented mechanical feedback path is shut off.

## Limits and next gate

This strong numerical result depends on a favorable setup: only **two** varied parameters, known and matching forward equations, clean synthetic observations, fixed pacing/initial state, and a smoothly interpolated 5×5 parameter grid. The 25 grid points are **not independent biological cells**. No model discrepancy, measurement noise, ion-channel variability beyond the one Ca parameter, pacing variation, external simulator, or experimental out-of-domain data was used. The largest beat-17→20 drift was 9 ms APD and 0.00656 µM Ca, so the score concerns a matched 20-beat protocol rather than proven steady-state physiology. The previously selected illustrative pair retained a large high-load separation after 40 beats, but this does not establish convergence for the whole grid.

Accordingly, the current claim is **a working reverse individualized point twin in a controlled rabbit closed-loop simulator**. It does not establish a robust rabbit population twin, a posterior over hidden physiology, alternans prediction, or a human bidirectional cell model.

The next decisive stress test should be frozen before generating new results: add a third independent variable such as Ca handling, use off-grid parameter draws and observation noise, require a beat-convergence criterion, and evaluate a new load/pacing challenge. Keep entire virtual cells disjoint between training and test. A direct nonlinear predictor must remain a baseline. If low+mid mechanistic personalization still improves high-load AP/Ca under those conditions, the reverse branch would have a much stronger method result.

## Artifacts

- Summary figure: `analysis/mct_feedback_audit/reverse_grid25_summary.png`.
- Scored data: `analysis/mct_feedback_audit/reverse_grid25_score.json`, `reverse_grid25_inverse_diagnostics.json`, `reverse_twin25_score.json`, and `reverse_calibration_audit_score.json`.
- Reproduction scripts: `analysis/mct_feedback_audit/run_mct_reverse_grid25.m`, `analysis/score_mct_reverse_grid25.py`, `analysis/fit_mct_reverse_twin25.py`, `analysis/mct_feedback_audit/run_mct_reverse_twin25.m`, `analysis/score_mct_reverse_twin25.py`.
- Calibration audit: `analysis/mct_feedback_audit/run_mct_reverse_calibration_audit.m`, `analysis/score_mct_reverse_calibration_audit.py`.
