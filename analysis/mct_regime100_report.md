# Rabbit MCT2024 mechanical-challenge regime audit

**Date:** 2026-09-29. **Scope:** simulated rabbit cardiomyocytes only. The frozen first-gate protocol is [`mct_regime100_protocol.md`](mct_regime100_protocol.md); scores and traces are under `mct_feedback_audit/reverse_robust3/regime100/`.

## Question and design

Does a uniform long mechanical challenge reveal stable 1:1, drifting, or period-2 AP/Ca responses, and is the cohort large enough to benchmark prediction of the late dynamical regime? We reused the existing 48 development and 16 off-grid test parameter vectors. Every vector was simulated for **100 beats** at high gel crosslinker concentration `CL=0.10`, with 1000-ms pacing and the same author-supplied initial state. APD is the time above −67 mV per beat; Ca is peak intracellular concentration. Labels were defined **before** this uniform run using beats 81–100 and strict last-10 criteria. Magnitudes are descriptive last-10 medians, not steady-state values for an alternating response.

## Frozen high-load gate

| Split | Stable 1:1 | Drifting or complex | Period-2 candidate |
|---|---:|---:|---:|
| Train, n=48 | 45 | 1 | 2 |
| Off-grid test, n=16 | 15 | 0 | 1 |
| Total, n=64 | 60 | 1 | 3 |

The predetermined classifier gate required **at least 5 train and 2 test period-2 cases**, plus 5 stable cases in each split. It **failed** (2 and 1 period-2 cases). Thus this cohort is a **regime-discovery map**, not a statistically useful supervised regime-prediction benchmark. No target labels were used to tune a predictor; no classifier was trained.

The three period-2 candidates were `train_016`, `train_036`, and `test_006`. Their last-10 odd/even mean gaps were, respectively, **95.2, 88.6, 99.6 ms** APD and **0.148, 0.139, 0.156 µM** Ca peak. All three passed the same frozen label after a separate stricter `ode15s` run (`RelTol 1e-6`, `MaxStep 0.5 ms` rather than `1e-5`, `1 ms`): APD gaps **96.6, 89.2, 100.4 ms** and Ca gaps **0.150, 0.142, 0.157 µM**. This supports repeatability under these two solver settings, while the model's stateful mechanics update inside the ODE callback prevents us from calling the pattern a mathematically established bifurcation.

## Beat 40 is a different target from the late response

Across all 64 high-load simulations, last-10 APD median minus beat-40 APD was **−29.5 ms** median (IQR −35.6 to −23.4 ms; range −63.5 to −11.0 ms). **55/64** differed by more than 20 ms. Ca peak changed by **−0.0215 µM** median (IQR −0.0315 to −0.0118 µM); 35/64 changed by more than 0.02 µM. Beat-40 outputs from the 40-beat and 100-beat reruns matched exactly for both scored APD and Ca peak, so the late shift is not due to changing simulation horizon in these runs. Most traces nevertheless satisfied the specified late 1:1 criterion by beat 100. The previous beat-40 point-prediction task therefore measured a transient and did not determine the late regime.

## Selected-background mechanism checks, explicitly exploratory

After seeing the three candidates, we selected **those same backgrounds** for two separate checks, keeping their CaL and SERCA scales fixed and retaining the original frozen regime labels. Selection on the outcome means these experiments do not estimate population prevalence or give an independent test set.

| Background | CL 0.02 | CL 0.06 | CL 0.10 | CL 0.10 with gain 0 or 0.7 |
|---|---|---|---|---|
| `train_016` | stable 1:1 | stable 1:1 | period-2 candidate | stable 1:1 at both gains |
| `train_036` | stable 1:1 | stable 1:1 | period-2 candidate | stable 1:1 at both gains |
| `test_006` | stable 1:1 | stable 1:1 | period-2 candidate | stable 1:1 at both gains |

Each low/middle-load and gain-ablation condition was run for the same 100 beats. At `CL=0.10`, the native gains are 1.231, 1.079 and 1.300, respectively. Thus the alternating response in these three **model parameter backgrounds** requires both the higher mechanical condition and stronger implemented feedback than gain 0.7. This is an intervention on the model's gel-force-dependent L-type Ca permeability factor; it does not identify a molecular cause in real cells, an exact critical gain/load, or an arrhythmia phenotype.

A waveform-level overlay of beats 99 and 100 for `train_036` shows that the high-load AP plateau/repolarization and Ca-transient amplitude genuinely differ between successive beats. At middle load the corresponding two voltage and Ca waveforms nearly overlap. Thus the period-2 feature is not only an APD threshold-crossing artifact. In this example the larger Ca transient coincides with the shorter APD; the precise current-level mechanism was not decomposed here.

## Interpretation and next gate

The strongest supported positive result is **mechanical-load-conditioned, feedback-dependent late AP/Ca alternation in three selected rabbit model variants**, with concordant stricter-solver reruns. The stronger general claim—that low+middle observations can predict which unseen high-load regime a new cell enters—remains **untested** because the original off-grid cohort has only one period-2 test case. The next separate study would need a newly specified cohort that deliberately samples the transition region, checks longer horizons and numerical implementation, and preserves a fully untouched off-grid test set. It should compare regime prediction against strong direct nonlinear baselines only after enough examples exist.

The human ToR-ORd/Land forward branch is a different species and coupling architecture; these rabbit MCT2024 results are not parameters for a unified human model. `CL` is an imposed gel crosslinker setting, not a directly measured force or sarcomere stress.

## Reproduction and artifacts

- Run: `analysis/mct_feedback_audit/run_mct_reverse_regime100.m`; score: `analysis/score_mct_regime100.py`.
- Selected checks: `analysis/mct_feedback_audit/run_mct_regime100_gain_ablation.m`, `run_mct_regime100_load_check.m`, and corresponding `analysis/score_mct_regime100_*` scripts.
- Numeric: `analysis/mct_feedback_audit/reverse_robust3/regime100/score.json`, `gain_ablation_score.json`, `load_check_score.json`.
- Figures: `regime_map.png`, `beat40_vs_late.png`, `late_beat_examples.png`, `gain_ablation_beats.png`, `load_check_beats.png`, `train036_waveform_overlay.png` in that same directory.
