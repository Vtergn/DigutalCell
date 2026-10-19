# Dynamic discrepancy benchmark: protocol lock before scoring

This is a **within-core, cross-feedback-form synthetic benchmark**, not an independent-published-model or real-human validation. The independent ORd–MechChem implementation did not pass a physically acceptable rapid-length-pulse test, so it is excluded from acute-pulse truth. The current task first asks whether dynamic inputs make simple extrapolation insufficient. A neural residual is introduced only if that gate is passed.

## Cells and physics families

Use all 64 individuals from the new-seed ToR-ORd/SAC–Land confirmation cohort, retaining its 13 heterogeneous parameters and prepaced states. The original pre-outcome plan used the first 24; the fixed QC rule left 17 complete cells, so the pre-outcome amendment below extended it to all 64. No cell is selected by response size. For each cell, generate the same length protocols under three equations:

1. **Prior:** published-inspired ToR-ORd/SAC–Land implementation already used in this project.
2. **Saturating SAC truth:** effective stretch input saturates (`sac_saturation=0.05`), anchored at λ=1.10; the remaining ODE is unchanged.
3. **No direct SAC truth:** set the direct stretch-sensitive ionic increment to zero while retaining dynamic TnC–Ca feedback and Land mechanics.

Families 2 and 3 are controlled modifications of the same source code. They do not count as independent physiological models.

## Mechanical protocols

All pulses use a 1 ms ramp and 1000 ms pacing cycle. Each form has an unperturbed baseline waveform. The two observed calibration challenges are:

- `C1`: early, onset 150 ms, λ peak 1.05, hold 10 ms.
- `C2`: late, onset 600 ms, λ peak 1.10, hold 10 ms.

The three sealed targets are:

- `T1`: onset 300 ms, λ peak 1.10, hold 100 ms (**unseen duration**).
- `T2`: onset 300 ms, λ peak 1.15, hold 10 ms (**unseen amplitude**).
- `T3`: compound early λ1.08/10 ms plus late λ1.10/20 ms (**unseen composition/history**).

The target waveform and all target-derived features are inaccessible to calibration, model selection, and the direct model's fitting. Source-family direct models may train on the original 64-cell cohort's five short-pulse design protocols, never on `T1`–`T3`.

## Outputs, baselines, gate

Save actual λ(t), voltage, free Ca, and active tension at 1 ms resolution. Primary score is waveform error in target pulse plus 200 ms after each pulse, separately for V, Ca, and T; report full-beat error secondarily. Compare at least:

- zero response relative to the cell's observed baseline;
- per-cell linear superposition of the two observed calibration pulse responses, shifted to target onset and scaled by peak Δλ (no fitted free curve);
- prior physical simulation after baseline/calibration parameter matching;
- a direct sequence predictor trained only on the original source-family design protocols;
- oracle true-form simulation (zero numerical target error, diagnostic only).

The **difficulty gate** is passed only if simple waveform transport/superposition shows a substantial, reproducible target-window error under at least one hidden truth family and the wrong mechanistic prior retains useful structure on at least one target. Otherwise the chosen task is either too easy or the prior too wrong for an honest residual-UDE demonstration. Do not tune protocol choices after scoring.

No regime-transition claim is permitted from one 1000 ms beat. A separate multi-beat adaptation protocol is required before reporting stable/drifting/period-2 classes.

## Pre-outcome QC amendment

The first 24-cell run completed 17/24 under all protocols; 7 cells failed the locked physical/numerical QC, primarily during the larger-amplitude target. **No target prediction errors had been scored at this point.** To avoid a 17-cell conclusion, extend the run to all 64 cells in the same independently seeded confirmation cohort. Keep all waveforms, protocols, parameter ranges, QC limits, and scoring definitions unchanged. Report both the initial 24-cell pass rate and final 64-cell pass rate. No replacement cells are sampled.
