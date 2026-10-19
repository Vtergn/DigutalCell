# Can C1/C2 recover physiology before feedback-form inference?

## Frozen benchmark and inference

The baseline, two 10-ms calibration probes, three sealed dynamic targets, source/development and new-seed/confirmation cells, and quality bounds are unchanged from the preceding dynamic tests. The inverse plan was recorded in [dynamic_physiology_inverse_protocol.md](dynamic_physiology_inverse_protocol.md) before these native ODE target predictions were scored. All results are **synthetic within a shared ToR-ORd/SAC–Land core**. They do not validate a human-cell digital twin against another published model or real recordings.

The original 64-cell development cohort provided 47 complete cells; the independent 64-cell confirmation cohort provided 45. Complete-case QC checks every required protocol, so these numbers exclude simulated cells that failed a target condition and do not characterize that excluded population. Five-fold development-cell CV compared ridge, RBF kernel ridge and PLS on 13 normalized physiological parameters, using either unperturbed V/Ca/tension or that baseline plus C1/C2 response differences. PLS with 20 components was selected for both observation sets **using development parameter error only**. The test cell's true parameter vector and 50-state initial condition were hidden from forward prediction. Each distinct inferred parameter vector was re-paced in the native equations (minimum 20, maximum 160 beats); 180/180 estimated vectors passed the fixed APD/Ca convergence check.

## Parameter and mechanism observability

| Observation for inverse model | Development CV mean normalized 13-parameter MAE | Confirmation original-SAC MAE | Confirmation feedback-strength MAE | Confirmation β1 MAE |
|---|---:|---:|---:|---:|
| Baseline V/Ca/T | .118 | .109 | .207 | .255 |
| Baseline + C1/C2 | .122 | .123 | **.015** | **.049** |

The confirmation parameter values in this table were used **only as diagnostics**, never as inverse inputs or model-selection targets. C1/C2 selectively constrain feedback-related parameters while slightly worsening average 13-parameter error. Under saturating-SAC truth, the C1/C2 inverse mean normalized error is .173; under no direct SAC it is .194, because this source-form inverse is exposed to model-form shift. The baseline-only inverse remains .109 across forms because the unperturbed traces are the same.

After native re-pacing, only the simulated baseline/C1/C2 responses of the inferred cell were used to estimate the direct-SAC gate and its C1/C2 amplitude contrast:

| Physiological state used | Direct-SAC presence AUC | Original vs saturating SAC AUC, 95% cell-bootstrap CI |
|---|---:|---:|
| Old nearest prototype | .991 | .529 [.508, .561] |
| Baseline inverse | 1.00 | .893 [.844, .945] |
| Baseline + C1/C2 inverse | 1.00 | **.919 [.851, .976]** |
| Actual same-cell physiological parameters (oracle diagnostic) | — | 1.00 |

The presence and form AUCs refer solely to these noiseless synthetic forms, with 45 independent confirmation *virtual cells* and paired mechanism conditions. Baseline versus baseline+C1/C2 form-AUC improvement is modest; no claim of a statistically established difference between those two inverse inputs is made. The large improvement over the old prototype supports the specific conclusion that **physiological-state error was masking a feedback-form signal already present in C1/C2**.

## Sealed native-ODE functional prediction

The table reports target-window, baseline-aligned **voltage waveform MAE in mV**, with target window defined as each pulse plus 200 ms. Baseline alignment uses the observed unperturbed waveform and never the target. Native ODE simulations were run at each inferred parameter vector, then with a gate fitted only from C1/C2.

| Hidden truth / target | Old prototype + gate | Baseline inverse + gate | Baseline+C1/C2 inverse + gate |
|---|---:|---:|---:|
| Original SAC / 100-ms hold | 5.27 | 3.03 | **1.38** |
| Original SAC / λ1.15 short pulse | .176 | .0558 | **.0151** |
| Original SAC / compound pulse | .311 | .158 | **.0537** |
| Saturating SAC / 100-ms hold | 5.24 | 5.32 | **2.63** |
| Saturating SAC / λ1.15 short pulse | .433 | .456 | .443 |
| Saturating SAC / compound pulse | .316 | .165 | **.139** |
| No direct SAC / 100-ms hold | .197 | .0539 | **.0170** |
| No direct SAC / λ1.15 short pulse | .0639 | .0213 | **.0063** |
| No direct SAC / compound pulse | .171 | .0966 | **.0369** |

The columns can contain slightly different QC-complete cells. Native pre-pacing succeeded for 180/180 inverse states. Native source-physics targets succeeded for 178/180 unique states; gate targets succeeded for 265/270 cell×form×inverse rows. The detailed V/Ca/tension scores, raw absolute waveform errors, cell IDs and failure flags are in `dynamic_inverse_target_scores.json`. For an actual paired comparison in original-SAC truth, baseline+C1/C2 inverse plus gate reduces long-hold V error relative to baseline-only inverse by **1.69 mV** (95% cell-bootstrap improvement 0.19–3.77 mV, 44 common cells). The same comparison in saturating-SAC truth points toward improvement but has a wide interval crossing zero.

Ca and tension predictions also generally improved with inverse-state estimation, but the effect is channel and protocol dependent. The model's different failures should not be reduced to a single combined waveform number: long-hold voltage remains notably inaccurate in the saturating truth despite high form-classification AUC.

## Re-test of the already fixed narrow flux residual

The earlier shared bounded SAC-increment slope and its C1/C2 activation threshold were applied **unchanged** after improved physiology inference. Both variants used exactly the same inferred parameters, re-paced state, gate, and target protocols; only the SAC-dependent ionic increment differed. The residual run had the same 265/270 success count as the gate run.

For **baseline+C1/C2 inverse** in saturating-SAC truth, the residual changed V MAE:

| Target | Gate only | Gate + fixed residual | Residual effect, paired 95% cell-bootstrap CI |
|---|---:|---:|---:|
| 100-ms hold | 2.627 | 2.639 | +.012 mV; no clear benefit |
| λ1.15 short pulse | .443 | **.164** | −.279 mV [−.332, −.227] |
| Compound | **.139** | .158 | +.019 mV [.0048, .0334] |

Under original SAC, the residual also slightly worsened the large short-pulse result (.0151→.0184 mV), while leaving no-direct-SAC predictions unchanged because the gate was near zero. The fixed correction therefore helps **amplitude extrapolation** of the saturating law but does not repair the longer history-dependent response. It fails the original multi-target go/no-go criterion for a general flux-residual UDE claim.

## Interpretation and boundary

This stage answers the immediate question positively but narrowly: **better cell-state inference can reveal feedback-form information and substantially improve sealed dynamic prediction.** The earlier prototype-level failure was not evidence that the two short probes lacked all form information. At the same time, 13-parameter recovery remains incomplete, especially outside the source feedback form, and a high form AUC does not guarantee accurate long-hold trajectories. The limited residual is not a neural UDE, and these results do not justify one yet.

The next physically meaningful uncertainty is whether the remaining long-hold error comes from imperfect physiological state, an inadequate SAC time/history law, or other feedback pathways. That requires an intervention that separates those possibilities or an independent model/real mechanically challenged cell for truth. Additional network capacity on the same C1/C2 pair would not by itself supply that information.

## Files and verification

- Inverse development and parameter diagnostics: `analysis/infer_dynamic_physiology.py`, `dynamic_physiology_inverse_score.json`.
- Native re-pacing and calibration simulations: `analysis/run_dynamic_inverse_calibration.m`, `dynamic_inverse_calibration.mat`.
- Calibration-only gate: `analysis/fit_dynamic_inverse_gate.py`, `dynamic_inverse_gate_score.json`.
- Native sealed-target prediction: `analysis/run_dynamic_inverse_targets.m`, `dynamic_inverse_target_scores.json`.
- Fixed residual retest: `analysis/prepare_dynamic_inverse_residual.py`, `analysis/run_dynamic_inverse_residual_targets.m`, `dynamic_inverse_residual_scores.json`.
- MATLAB RHS source invariance and zero-stretch residual invariance passed in `analysis/check_flux_invariance.m`.
