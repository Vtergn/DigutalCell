# Off-grid three-parameter reverse-twin stress test: observability persists, robustness gate fails

Date: 2026-09-29. The frozen design and decision rule are in [`mct_reverse_robust3_protocol.md`](mct_reverse_robust3_protocol.md). All results are from the **rabbit MCT2024 closed-loop simulator**. The human ToR-ORd→Land platform remains a separate, one-way model domain.

## Design and engineering checks

We varied three parameters: the gel-stress→L-type Ca feedback gain, basal L-type Ca permeability, and maximal SR Ca uptake (SERCA). Forty-eight training and sixteen disjoint off-grid test parameter vectors were independently sampled by fixed-seed Latin hypercubes. Every variant was simulated at `CL=0.02, 0.06, 0.10` for 40 beats at 1000-ms pacing, giving 192 native-ODE conditions. Each test cell's high-load APD/Ca was hidden during individualization. APD is the time above −67 mV, **not APD90**; `CL` is a gel setting, **not a measured force trajectory**.

The separate three-parameter MATLAB runner reproduces the prior reference exactly at unity parameter scales: maximum differences in saved time, voltage, Ca, and gel-stress arrays were all zero. A prespecified four-test-identity calibration audit sent the inverse parameters back through the native ODE at their *observed* loads. Low+middle calibration gave mean APD error 0 ms and Ca-peak error 0.00043 µM. Thus the clean inverse fits in those audited cases are not merely artifacts of an inaccurate calibration emulator.

## What survived the harder test

Mechanical challenge still resolved a hidden feedback parameter. Mean absolute MCT-gain recovery error across the 16 off-grid test vectors fell from **0.382 with low load alone** to **0.028 with low+middle load**. Basal L-type Ca scale error fell from 0.0318 to 0.00183; SERCA scale error fell only from 0.0392 to 0.0323. Under the prespecified observation-noise stress (2 ms APD and 0.01 µM Ca SD), corresponding errors were 0.117, 0.0160, and 0.104. The challenge therefore adds information about feedback gain, while Ca-handling uncertainty remains substantial and noise-sensitive.

At the fixed 40th beat, the native-ODE twin improved markedly *relative to its own low-only calibration*:

| Clean high-load predictor | APD MAE (ms) | Ca-peak MAE (µM) |
|---|---:|---:|
| Native twin, low only | 27.00 | 0.08475 |
| Native twin, low + mid | **2.06** | 0.00686 |
| Direct ridge, low + mid | 12.17 | 0.03856 |
| Direct nonlinear RBF, low + mid | 2.13 | **0.00260** |
| Per-cell linear load extrapolation | 25.00 | 0.08081 |

The mechanistic twin no longer has a meaningful predictive advantage over the strongest direct baseline: their clean APD errors differ by only 0.065 ms, and direct RBF is about 2.6× better for Ca peak. The native twin beats direct RBF in 10/16 test variants for APD but only 5/16 for Ca. The previous 25-point, two-parameter advantage therefore does **not** survive this tougher comparison as a general claim.

Across three fixed noisy-observation draws, the native twin's mean MAE was **8.27 ms and 0.02355 µM**, while noise-augmented direct kernel ridge achieved **7.33 ms and 0.02206 µM**. Winners vary by draw and output. These arbitrary synthetic noise levels are a computational stress test, not a measured assay-noise model. The noise result does not support a robust mechanistic-prediction advantage.

## The predeclared convergence gate failed

The protocol required the last-four-beat APD range to be at most 5 ms and Ca-peak range to be at most 1% in **every** condition. Six of 192 conditions failed, including one test cell's middle-load calibration. The largest 40-beat-window APD range was 6 ms and relative Ca range 1.38%. Consequently, the 40th-beat scores above describe a matched transient simulation protocol and **cannot be presented as converged-state predictive performance**.

We extended only the six failing conditions to 60 beats as a diagnostic; this selective check does **not** satisfy the predeclared all-condition 60-beat rerun. Five had smaller local beat-to-beat fluctuations by beat 60, yet their beat-60 APD differed from beat 40 by roughly 23–27 ms and Ca peak by 0.035–0.043 µM. This shows that a short local stability window can miss slower drift. The remaining condition (`train_036`, `CL=0.10`) developed much larger alternating APD/Ca responses.

For that **single model variant**, a 100-beat run showed last-ten-beat APD alternating around 340 and 252 ms and Ca peaks around 0.623 and 0.761 µM. Repeating the run with `ode15s` relative tolerance tightened from `1e-5` to `1e-6` and maximum step halved from 1 to 0.5 ms gave an APD odd/even gap of 89.2 versus 88.6 ms under the original settings; corresponding Ca gaps were 0.142 versus 0.139 µM. This supports a solver-setting-insensitive **model alternation candidate** in that one parameter/load condition. It does not establish a biological arrhythmia, an alternans prevalence, or independence from all other numerical/model assumptions.

## Decision and scientific meaning

**Predeclared robustness gate: NO-GO.** The clean prediction is accurate within the fixed 40-beat same-equation task, but the direct nonlinear baseline matches or beats it, noisy superiority is absent, and the convergence requirement fails. We therefore do not upgrade the reverse branch to a robust rabbit electromechanical digital twin.

The positive result is narrower and still useful: **a middle mechanical challenge improves observability of the implemented feedback gain**, even after adding a third Ca-handling parameter and moving test variants off the original grid. A separate dynamic finding is that a fixed-beat endpoint may hide a later transition into alternating AP/Ca responses. Both are model-internal results; their relevance to real rabbit or human myocytes remains untested.

The next experiment should define the outcome as a *beat-resolved dynamic regime* as well as APD/Ca magnitude. It needs a longer common adaptation protocol, an explicit long-window drift and period-two criterion, and held-out high-load trajectories. The point twin and direct nonlinear predictor should then be compared on both magnitude and regime prediction. A sensitivity check on the original model's stateful global mechanics update would also be needed before interpreting the alternating trajectory mechanistically.

## Artifacts

- Cohort design and native conditions: `analysis/mct_feedback_audit/reverse_robust3/design.csv`, `design_manifest.json`, `native_cohort/`.
- Numeric outputs: `cohort_score.json`, `inverse_diagnostics.json`, `direct_predictions.json`, `twin_score.json`, `calibration_audit_score.json`, `failed60_score.json`, `train036_high100_score.json`, `train036_solver_sensitivity.json` in the same folder.
- Figures: `robust3_summary.png` and `train036_high100_beats.png` in the same folder.
- Main scripts: `analysis/prepare_mct_reverse_3param.py`, `analysis/design_mct_reverse_robust3.py`, `analysis/score_mct_reverse_robust3_cohort.py`, `analysis/fit_mct_reverse_robust3.py`, `analysis/score_mct_reverse_robust3_twins.py`. MATLAB drivers are in `analysis/mct_feedback_audit/`.
