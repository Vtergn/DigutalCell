# Exploratory timing grid using the published human ToR-ORd+SAC model

Date: 2026-10-01. The first 600-ms acute-stretch run has already been observed; this timing extension is exploratory and is not a prospective validation.

Reuse the 120-beat SAC-specific state from `human_sac_reference_pilot/prepace_state.mat`. This state passed an early last-10-beat threshold but later proved not fully adapted: APD shifts 7 ms by beat 500. The 500-beat audit is therefore the preferred result for final interpretation; the 120-beat timing grid remains exploratory. With the author's default healthy-current parameter vector, common state and 1000-ms pacing, apply 10-ms step stretches at 150, 300 or 600 ms into the next beat. At each onset run λ = 1.00, 1.05, 1.10. Restart `ode15s` at the stimulus on/off boundaries and save the complete Vm/Ca trace. Comparisons must use the λ=1.00 trace at the **same onset** to separate numerical segmentation effects from the SAC effect.

Report APD above −67 mV, Ca peak and post-stimulus voltage/Ca excursions, and check whether a second action potential appears. These are model signatures. No Land force or reverse human feedback parameter is fit in this phase, and the published SAC model uses a mixture of human and animal experimental constraints rather than a fully direct human-ventricular calibration.
