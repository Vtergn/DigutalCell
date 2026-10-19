# Frozen robustness gate for the rabbit reverse point twin

Written 2026-09-29 before generating any three-parameter outcomes. This is a development gate within the native MCT2024 rabbit simulator, not experimental validation.

## Design

- 48 training and 16 disjoint test parameter vectors, independently sampled by fixed-seed 3D Latin hypercubes. Ranges: MCT feedback gain `0.20–1.30`, basal L-type Ca permeability scale `0.90–1.10`, maximal SR Ca uptake (SERCA) scale `0.85–1.15`.
- Each vector is simulated at gel settings `CL=0.02, 0.06, 0.10`, starting from the same author-supplied initial state for 40 beats at 1000-ms pacing. No variant's high-load output is used as its own calibration input.
- Inputs: final-beat APD above −67 mV and Ca peak at low load, then low+middle load. Targets: clean final-beat APD and Ca peak at high load.
- The primary split is by parameter vector, not by waveform or load. All 16 test vectors are outside the prior 5×5 grid and outside the 48 training identities.

## Convergence check

For every condition, inspect beats 37–40. A condition passes if APD changes by at most 5 ms and Ca peak changes by at most 1% of its beat-40 value across that window. If any condition fails, the 40-beat prediction results remain descriptive; an all-condition 60-beat rerun is required before calling this robustness gate passed. APD is the duration above −67 mV, **not APD90**.

## Prediction rules

- Direct baselines: no-change; per-cell linear extrapolation `high=2×mid−low`; train-only standardized ridge (penalty 1); train-only standardized thin-plate RBF (smoothing `1e-6`). Low and low+mid versions of ridge/RBF are evaluated where applicable.
- Mechanistic point twin: fit thin-plate RBF **forward emulators only for the available low and middle calibration features** using the 48 training parameter vectors. Optimize the three parameters within the prespecified box against each test cell's observed features. The high-load prediction is then obtained from the **original three-parameter native MCT ODE**, not from the calibration emulator.
- For clean calibration, compare low-only and low+mid. The main clean target is test-set APD/Ca MAE at high load.
- As a native-equation calibration audit, preselect test identities `test_000`, `test_005`, `test_010`, and `test_015` before scoring. Re-run their inferred clean low and clean low+mid parameter vectors at the observed loads and report calibration residuals; this checks whether the inverse emulator is exploiting its own approximation error.
- Secondary noise stress: three fixed-seed independent Gaussian observation-noise draws for each test vector, with SD 2 ms APD and 0.01 µM Ca at the observed low/middle loads. The high-load truth remains the clean native-ODE output. These SDs are arbitrary computational stress levels and do not claim a measured assay noise model. Direct noisy-input comparators are additionally trained using eight synthetic noise augmentations per training vector, with no target/test data.

## Decision

A positive **development** result requires all conditions to pass the 40-beat convergence check; clean low+mid native-ODE twin MAE must improve over clean low-only twin in both APD and Ca and beat the best direct baseline in at least one output without materially losing the other. Under noise, report all three draws and mean errors; no result is called robust if the advantage vanishes or only a cherry-picked draw wins. Parameter recovery is diagnostic, not the primary endpoint. No p-value is interpreted as a biological-population inference.

This same-family test does not assess model misspecification, pacing generalization, alternans, or human-cell transfer.
