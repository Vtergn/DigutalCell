# Long-hold tail: local dynamical-sensitivity pilot

2026-10-02. Exploratory analysis in the existing human ToR-ORd/SAC–Land
synthetic model. The protocol was written before these scans in
`dynamic_event_boundary_protocol.md`. This is a targeted audit of three
previously identified high-error cells, not an independent cohort or a new
biological validation.

## Question and experiment

Could the long-hold voltage tail reflect sharp sensitivity of the second
stretch-evoked voltage response to a small error in task-relevant physiology?
For each cell, only one previously implicated parameter was interpolated
between its inferred and known synthetic true value. All runs received the
same stretch: onset 300 ms, length ratio 1.10, hold 100 ms, with the
saturating-SAC law. The acute scan kept the inferred pre-paced 50-state vector
fixed. The main event measure was time above -50 mV in 300–602 ms. This
threshold is an operational waveform descriptor, not a physiological event
label or proof of a bifurcation.

## Results

| Cell, varied parameter | Narrow parameter flank | Acute time above -50 mV | After 60-beat re-pacing | Interpretation |
|---|---:|---:|---:|---|
| 8, INaL scale | 0.985123 → 0.986954 | 225 → 155 ms | 227 → 146 ms | Large local change persists after re-pacing. |
| 43, IKr scale | 0.870553 → 0.873490 | 142 → 200 ms | 148 → 200 ms | Large local change persists after re-pacing. |
| 57, effective ionic stretch gain | 1.130066 → 1.129325 | 208 → 170 ms | 123 → 120 ms | Acute contrast nearly disappears after re-pacing; initial-state dependence dominates this narrow flank. |

Thus the convincing positive finding is **two of three preselected tail cells
show a sharp local change in second-response duration under a very small
one-parameter perturbation, even after re-pacing**. The third cell is a useful
counterexample: fixed-state sensitivity alone can suggest a misleading
boundary. It should not be pooled with cells 8 and 43 as evidence of a stable
parameter boundary.

The voltage waveform comparison gives another check that the duration metric
is not acting alone. On the same acute flanks, the peak voltage changes from
approximately +9.8 to -14.4 mV in cell 8, and -15.9 to +15.5 mV in cell 43.
The flanking trajectories start close together, then separate during the
second stretch response. This is compatible with nonlinear state-dependent
amplification, but these plots do not identify which ionic current is the
causal switch. Current values sampled at a single time cannot establish
causality.

Numerical refinement used relative/absolute tolerances of 1e-7/1e-9 and a
0.5 ms maximum step, compared with 1e-6/1e-8 and 1 ms in the default run.
At all six acute flank points, event durations were unchanged. Across the
300–602 ms voltage window, default-versus-strict mean absolute differences
were 0.0016–0.0175 mV. The final pre-paced state changed by a relative RMS
of roughly 1e-5 between the last two beats; this is reassuring, but it is not
a formal asymptotic convergence proof.

## What the pilot establishes and what it does not

The model has a narrow **predictive regime boundary candidate** for at least
cells 8 and 43 under this long-hold challenge. A small parameter error can
produce a much larger waveform error after the mechanical stimulus. This
offers a concrete physical target for future risk-aware personalization: the
relevant quantity may be a cell's margin to a long-hold response change, not
its average 13-parameter error.

The scan follows an oracle direction toward known synthetic truth, uses only
three cells selected from prior tail analysis, and varies one parameter at a
time. It does not prove a mathematical bifurcation, a new human-cell mechanism,
or that a neural model can detect the margin from C1/C2 observations. The
re-paced behavior of cell 57 shows why those claims would be premature.

The next go/no-go test is a newly generated, sealed cohort: define the
long-hold response descriptor and margin in a development set, then ask
whether C1/C2-based uncertainty predicts high-error cells and whether a
cell-specific extra probe reduces held-out long-hold tail risk. Any learned
representation must beat direct risk scores and simple posterior or local
sensitivity baselines. This pilot alone does not justify a PINN, UDE, or
higher-order network.

## Reproducibility

- Simulation: `run_dynamic_event_boundary_pilot.m` modes `acute`, `refine`,
  `repace`, `repace_refined`, `strict`, and `waveform`.
- Tabular outputs: `human_bidirectional_output/dynamic_event_boundary_*.csv`.
- Plot code: `plot_dynamic_event_boundary_pilot.py`.
- Figures: `human_bidirectional_output/dynamic_event_boundary_pilot.png` and
  `human_bidirectional_output/dynamic_event_boundary_waveforms.png`.
