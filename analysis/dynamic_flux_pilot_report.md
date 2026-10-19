# C1/C2-only SAC flux correction: strict pilot result

## Question and design

Can baseline plus two brief mechanical probes individualize an uncertain SAC feedback law well enough to predict sealed dynamic interventions? The selection and scoring rules were frozen in [dynamic_flux_pilot_protocol.md](dynamic_flux_pilot_protocol.md) before target scores were read. This is a **within-core synthetic benchmark**: all three truth families share ToR-ORd/Land and differ only in direct SAC feedback (original, saturating, absent). It is neither independent-model nor real-cell validation.

The development and confirmation cohorts had separate random seeds and original 13-dimensional parameters. Fixed full-protocol QC retained 47/64 development cells and 45/64 confirmation cells. One of the 135 cell×truth combinations failed QC under the inferred gate and under gate+residual, so the comparisons involving those methods use the same 44 valid cells for that family.

## Narrow model actually tested

The test cell's true parameters and pre-paced state were hidden. Its unperturbed AP/Ca/tension waveform selected one of 47 development-cell physiological prototypes. Two observed 10-ms pulses (C1 early λ1.05; C2 late λ1.10) then estimated a nonnegative per-cell SAC gate. A shared, bounded correction slope was learned from the separate development cohort's C1/C2 responses; the test cell's C1/C2 ratio decided whether that correction was active. No sealed target waveform was used to choose a prototype, gate, or slope.

Only the complete SAC-dependent ionic increment in the native 50-state ODE was altered. The correction is zero at λ=1, bounded, and causal through the existing ODE states. **It is a one-parameter functional residual pilot, not a neural UDE/PINN.** The three native ODE variants evaluated were (i) physiological prototype alone; (ii) prototype+SAC gate; (iii) prototype+gate+shared/individualized flux slope. The large zero-response, transport, direct-ridge and oracle comparisons are in [dynamic_discrepancy_report.md](dynamic_discrepancy_report.md).

## Sealed-target results

Target-window waveform MAE below uses **baseline alignment**: observed test-cell unperturbed waveform + (prototype target ODE − prototype baseline ODE). This avoids counting a prototype's unperturbed AP/Ca offset as mechanical-feedback error. Raw ODE errors are also in `dynamic_flux_pilot_scores.json`; they are generally larger. Values are V (mV) / free Ca (µM) / tension (kPa).

| Hidden truth / target | Prototype physics | + SAC gate | + gate and flux residual |
|---|---:|---:|---:|
| Original SAC / long hold | 8.75 / .0218 / .205 | 5.27 / .0178 / .124 | 5.29 / .0178 / .122 |
| Original SAC / large short pulse | .471 / .0035 / .0439 | **.176** / .0034 / .0439 | .271 / .0034 / .0439 |
| Original SAC / compound | .504 / .0026 / .113 | **.311** / .0023 / .113 | .344 / .0023 / .113 |
| Saturating SAC / long hold | 8.87 / .0225 / .207 | 5.24 / .0175 / .120¹ | 5.27 / .0175 / .120¹ |
| Saturating SAC / large short pulse | .431 / .0034 / .0438 | .433 / .0036 / .0439¹ | **.286** / .0034 / .0438¹ |
| Saturating SAC / compound | .539 / .0027 / .113 | **.316** / .0024 / .114¹ | .344 / .0024 / .114¹ |
| No direct SAC / long hold | 8.39 / .0282 / .181 | **.197** / .0057 / .0388 | .197 / .0057 / .0388 |
| No direct SAC / large short pulse | .910 / .0050 / .0443 | .0639 / .0032 / .0442 | **.0582** / .0031 / .0442 |
| No direct SAC / compound | 1.08 / .0043 / .113 | **.171** / .0021 / .113 | .174 / .0021 / .113 |

¹ Gate and residual models have 44 rather than 45 successful cells in this truth family. The prototype column retains 45; paired differences below use the common 44. Single-target bold values identify the lowest listed V error and do not imply overall method superiority.

The critical paired comparison is **residual minus gate**. For saturating-SAC truth it improves the large short-pulse V MAE by **0.148 mV** (44 cells; cell-bootstrap 95% CI 0.080–0.228), but worsens the compound V MAE by **0.0276 mV** (CI 0.0134–0.0418); the long-hold change is not clear. On original SAC, the same residual worsens the large short pulse by **0.0954 mV** (CI 0.0498–0.1468) and the compound pulse by **0.0334 mV** (CI 0.0176–0.0492). These are paired cell bootstraps, not independent biological replicates.

## What is identifiable?

The C1/C2-derived gate distinguishes direct SAC present versus absent with **AUC 0.991** in the confirmation cohort. Its median is 1.10 (original), 1.13 (saturating), and 0.0034 (absent). It does not reliably distinguish *which present-SAC law* holds: saturation-form AUC from the inferred slope is **0.529**, near chance.

An oracle diagnostic reveals the source of this failure. If the test cell's true physiological parameters are supplied, the same-cell C1/C2 effective-gain medians are [1.00, 1.00] for original SAC and [1.41, 1.03] for saturating SAC; the saturation contrast has AUC 1.00 in these noiseless equations. Therefore the forms are distinguishable in principle under the simulated probes, but **imperfect physiological individualization masks the small form signal**. This oracle result is not a deployable twin score.

## Decision and limits

Under the pre-specified criterion, **the tested flux residual is NO-GO as an algorithm claim**: it improves one sealed target but not multiple dynamic targets and harms the original-SAC family. The **mechanism-presence gate is GO as a synthetic proof of principle**, particularly for the no-direct-SAC family. A larger neural residual is not justified by these data yet; it could fit the compensation between wrong physiological prototype and wrong SAC form rather than learn a transferable flux law.

The next decisive improvement is C1/C2-based physiological state inference (a posterior or better inverse model), followed by the *same locked* target test of the gate and residual. The prototype baseline has large long-hold error even under original-SAC truth, so parameter/state error is currently the dominant bottleneck. A genuinely independent published model and real paired mechanical challenge data are still required before broad claims about human-cell feedback.

## Reproduce

Run `generate_dynamic_discrepancy_cohort(64,'development')` for the separate development cohort, then `prepare_dynamic_flux_pilot.py`, `run_dynamic_flux_pilot.m`, `score_dynamic_flux_pilot.py`, `summarize_dynamic_flux_pilot.py`, and `diagnose_dynamic_gate.py`. Code and outputs are under `analysis/` and `analysis/human_bidirectional_output/`. The original RHS is unchanged when new feedback fields are omitted; the correction is isolated in `analysis/human_bidirectional_rhs.m`.
