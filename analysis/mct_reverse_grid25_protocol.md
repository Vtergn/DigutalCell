# Predefined exploratory reverse challenge benchmark

Date: 2026-09-29. This protocol was written while the 25-variant simulation was running, before scoring its new middle-load observations.

## Domain and scope

Use the native rabbit Shannon–Bers/Negroni MCT2024 model. The controlled parameters are MCT feedback gain (0.25, 0.50, 0.75, 1.00, 1.25) and basal L-type Ca permeability scale (0.90, 0.95, 1.00, 1.05, 1.10). Their Cartesian product gives 25 model variants. Each variant is run at gel crosslinker `CL=0.02, 0.06, 0.10` for 20 beats at 1000-ms pacing, always from the same author-supplied initial state. Low/high runs from the previous pilot are reused, and the missing runs are generated using exactly the same MATLAB runner.

These are **designed parameter grid points**, not randomly sampled cells. All reported prediction results are a development benchmark, not a biological population estimate or a validated AI method.

## Question

Does observing an intermediate mechanical challenge add information about the unseen high-load AP/Ca response beyond the low-load state?

## Frozen inputs, targets, and split

- Low-only input: final-beat APD above −67 mV and Ca peak at `CL=0.02`.
- Low+mid input: those two low-load features plus the same two features at `CL=0.06`.
- Blind target: final-beat APD above −67 mV and Ca peak at `CL=0.10`.
- Leave-one-**variant**-out evaluation. The held-out variant's low and middle responses are permitted as individual calibration observations; its high response is never used during fit. All outputs from that variant are excluded from population-model training.
- Report separate mean absolute error in milliseconds and µM. Also report paired per-variant improvements and median errors; no p-value is interpreted as a biological population inference.

## Fixed baselines

1. No-change control: predict high-load outputs using the latest observed load (`low` for low-only, `mid` for low+mid).
2. Direct ridge regression: train-only standardization, ridge penalty 1 in standardized feature/target space, fixed before results. Compare low-only and low+mid.
3. Per-variant linear load extrapolation: because the three `CL` settings are equally spaced, predict `high = 2 × mid − low` independently for APD and Ca. This is a strong physics-free challenge baseline.
4. Mechanistic individualization: fit a thin-plate radial-basis forward emulator of the **calibration outputs only** as a function of the two controlled parameters, excluding the held-out variant. Standardize outputs on training variants, use RBF smoothing `1e-6`, and minimize squared standardized calibration mismatch within the fixed parameter box. Fit once with low-only observations and once with low+mid observations. Run the **actual native ODE** at high load using each inferred parameter vector; the held-out variant's high response never enters calibration. Compare both with direct ridge and per-variant linear load extrapolation. Because all data share one deterministic simulator and a small two-parameter grid, this is a feasibility check, not a biological twin-validation study.

## Quality checks

- Inspect beat-17→20 drift; this is a fixed 20-beat protocol, not a guarantee of exact steady state.
- Verify reference parameter values reproduce the existing native MCT case. This was already checked for gain 1 and Ca-permeability scale 1: high-load voltage, calcium, time, and stress arrays were identical.
- Report `CL` as a gel setting rather than a measured force trajectory. Do not transfer fitted rabbit coefficients to the human ToR-ORd/Land platform.
