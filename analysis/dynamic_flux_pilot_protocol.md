# C1/C2-only narrow flux correction pilot: analysis lock

This document precedes scoring the pilot's sealed target waveforms. The prior dynamic gate already showed why dynamic model discrepancy is worth testing. The goal here is to remove test-cell true parameters, not to claim a mature UDE.

**Development / confirmation:** 64-cell source-seed cohort is the development set; the independent 64-cell confirmation cohort is the test set. Use only cells passing the fixed full-protocol QC for paired scoring; report attrition. No test-cell parameter vector, initial state or target waveform may enter matching, gate fitting or residual fitting.

**Observations for test-cell individualization:** baseline plus early λ1.05/10ms C1 and late λ1.10/10ms C2 AP/Ca/tension. Find one source-family virtual-cell prototype from baseline AP/Ca/tension at 10ms resolution, using channel scales estimated on the development cohort. That prototype supplies the 13 parameters and prepaced 50-state initial condition. This is a simple nearest-prototype personalization baseline, not a fully optimized posterior twin.

**Narrow correction:** all 50 original state equations stay in place. The only correction multiplies the complete SAC-dependent ionic state increment, preserving the original ionic bookkeeping:

`de = de(lambda=1) + gate_i * exp(beta_i*(0.1-|lambda-1|)) * [de_prior(lambda)-de(lambda=1)]`.

The increment is exactly zero at λ=1. The gain is clipped to [0,3]. It is causal and bounded; history enters through the original ODE states, not an extra neural state. This is a **parametric flux-residual precursor**, not a neural UDE. Its shared slope is derived only from the *development* cohort's C1/C2 saturation responses. Test-cell C2 determines a nonnegative SAC gate; the C1/C2 ratio decides whether the discrepancy slope is active. The exact fitting rules and thresholds are in `prepare_dynamic_flux_pilot.py` and frozen before target scoring.

**Sealed targets and comparisons:** T1 long hold, T2 λ1.15 short pulse, T3 compound early/late pulse. Compare zero response, causal transport, source-family direct ridge sequence predictor, physics-only individualized prototype, prototype plus mechanism gate, and prototype plus gate and flux residual. A known-form prototype selector is diagnostic upper bound only. Score full V/Ca/T waveforms in the locked pulse-plus-200ms windows, with whole beat secondary. Report failures and per-family results. No protocol or fitting threshold is adjusted based on sealed target scores.

Report both raw ODE waveform error and a baseline-aligned response score: observed test baseline + (prototype target ODE − prototype baseline ODE). Baseline alignment uses only the observed unperturbed waveform and removes the prototype's baseline offset; it does not use target outcomes. The latter is the main comparison for the incremental SAC correction, while raw error documents absolute twin fidelity.

**Interpretation gate:** A flux residual is promising only if its benefit over *the same individually selected physics prototype and gate* survives on independent cells and improves more than one dynamic target without materially worsening the short-amplitude target. Otherwise the fair conclusion is that C1/C2 do not sufficiently identify this residual under the present prior/model family. This synthetic within-core result cannot be advertised as independent-model or real-cell validation.
