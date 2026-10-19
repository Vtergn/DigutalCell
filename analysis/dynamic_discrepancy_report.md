# Dynamic mechanical challenge: model-discrepancy gate

## Scope and protocol

The protocol and QC rules were locked in [dynamic_discrepancy_protocol.md](dynamic_discrepancy_protocol.md) before target errors were scored. This is a **within-core, cross-feedback-form synthetic test**, not an independent human-model or experimental validation. One beat was simulated for each of 64 new-seed 13-parameter virtual cells under the source ToR-ORd/SAC–Land form, saturating SAC, and no direct SAC. Each cell received baseline, two observed 10-ms pulses (early λ1.05 and late λ1.10), and three sealed targets: a 100-ms hold, a λ1.15 pulse, and a compound early/late pulse. Waveforms comprise V (mV), free Ca (µM), and active tension (kPa) at 1-ms resolution.

The locked QC retained 17/24 pilot cells and 45/64 final cells for paired evaluation across all forms. Nineteen cells failed a physiological/numerical bound in at least one required condition. This is complete-case evidence; it must not be generalized to unstable or high-response cells.

## Target-window waveform MAE

Values below are averages over 45 cells. The target window includes the imposed pulse and 200 ms after its end; for the compound protocol, it includes both such windows. Each entry gives **V / Ca / tension** MAE.

| Hidden truth / target | Zero response | Causal short-pulse transport | Source physics with matching true parameters | Source physics plus gain fit from C1/C2 | Direct ridge sequence |
|---|---:|---:|---:|---:|---:|
| Saturating SAC / long hold | 7.52 / .0242 / .191 | 7.19 / .0230 / .190 | **.617 / .0013 / .0030** | 1.81 / .0019 / .0030 | 7.19 / .0225 / .188 |
| Saturating SAC / larger amplitude | .583 / .0074 / .206 | **.202 / .0034 / .205** | .419 / **.0007 / .0008** | .591 / .0008 / .0008 | .209 / .0021 / .0753 |
| Saturating SAC / compound | 1.40 / .0068 / .550 | 1.41 / .0070 / .550 | **.104 / .0002 / .0012** | .128 / .0002 / .0013 | 1.14 / .0054 / .453 |
| No direct SAC / long hold | **.120 / .0094 / .141** | .123 / .0095 / .141 | 7.99 / .0214 / **.0605** | .598 / .0141 / .0606 | .384 / .0095 / .139 |
| No direct SAC / larger amplitude | **.059 / .0052 / .208** | .057 / .0049 / .208 | 1.01 / .0037 / **.0040** | .135 / .0028 / .0040 | .592 / **.0029** / .0754 |
| No direct SAC / compound | .198 / .0040 / .557 | **.192 / .0031** / .556 | 1.12 / .0034 / **.0078** | .164 / **.0026** / .0079 | .647 / .0041 / .466 |

Source-physics truth is an exact same-equation positive control and has zero error by construction. The oracle true-form simulator likewise has zero error and is only a diagnostic. Source physics in the table is **given the actual cell parameters**; it is a favorable mechanism baseline, not a deployable individualized twin. The calibrated variant estimates only one nonnegative gain per observed channel from C1/C2, with the true parameter vector still given.

Cell bootstrap (10,000 resamples) shows that in saturating-SAC truth, source physics improves V MAE over causal transport by 6.58 mV (95% CI 4.11–9.81) for long hold and 1.30 mV (1.19–1.42) for compound challenge. For the larger-amplitude 10-ms pulse, the source physics is *worse* in V by .217 mV (.172–.264); transport does better there. The physical prior is therefore useful but not uniformly correct. Its V benefit on long/compound targets remains with calibration-only gain fitting (5.39 mV and 1.28 mV versus transport, respectively).

## Decision

The **task-difficulty gate passes for saturating-SAC truth**: two short pulses do not determine the long-hold or compound V/Ca/tension waveform by naive response transport, whereas the imperfect dynamical equations preserve substantial structure. The larger-amplitude short pulse is a counterexample: its V response is better predicted by simple transport. For no-direct-SAC truth, even a zero-response baseline is competitive for V; the wrong direct-SAC prior has a large systematic bias. A universal residual learned across these forms would hide this model-selection problem.

These findings justify a *narrow* next experiment on the missing SAC feedback flux, with shared discrepancy and low-dimensional cell correction. They **do not establish that a PINN/UDE will win**. Before claiming that, replace oracle parameter matching with C1/C2-only individualization, hold out whole cells and whole waveform families, and compare a flux-residual model with these strong protocol-aware baselines. A residual should be anchored to zero at λ=1, causal and bounded; Ca/charge bookkeeping should be checked. The current one-beat dataset cannot establish late dynamic regimes or recovery. Independent published-model truth and real mechanically challenged human-cell data remain absent.

## Reproduce

Run `generate_dynamic_discrepancy_cohort(64)` in MATLAB with `analysis` on the path, then `analysis/score_dynamic_discrepancy.py` and `analysis/summarize_dynamic_discrepancy.py` using the project `.venv_tcn` Python. Raw waveforms and scores are in `analysis/human_bidirectional_output/dynamic_discrepancy_cohort.mat` and `dynamic_discrepancy_scores.json`. The direct ridge uses only source-cohort short-pulse labels and fixed PCA/ridge settings, so its poorer performance on unseen pulse duration is a limitation of that baseline, not proof that all direct sequence models fail.
