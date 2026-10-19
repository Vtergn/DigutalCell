# Predictive-disagreement design: probe-library feasibility gate

Frozen 2026-10-03 before generating any new probe responses. The 64-cell
development cohort is the only population used for the library screen. This
stage does not fit an acquisition function or inspect a confirmation target.

## Question

Can this human ToR-ORd/SAC–Land system supply a **simulator-QC-safe** and
**response-diverse** set of mechanical probes? If not, a cell-specific
selection policy has no meaningful action space. “Safe” here means only that
the simulated V/Ca/tension trajectory meets the existing numerical and
physiological range checks; it is not experimental safety.

## Frozen candidates

Each pulse is `[onset_ms, lambda_peak, hold_ms, ramp_ms]`; all use a linear
ramp on and off, no overlap, and the existing 1000-ms pacing cycle. The
candidate list is:

| ID | Pulse sequence | Reason for inclusion |
|---:|---|---|
| 1 | `[600,1.03,10,1]` | small late load |
| 2 | `[600,1.05,10,1]` | moderate late load |
| 3 | `[600,1.08,10,1]` | larger late load |
| 4 | `[600,1.10,20,1]` | prior late C3 reference |
| 5 | `[600,1.10,35,1]` | prior duration comparison |
| 6 | `[600,1.08,35,1]` | amplitude-duration tradeoff |
| 7 | `[600,1.05,50,1]` | longer but weaker hold |
| 8 | `[600,1.08,10,10]` | slower ramp |
| 9 | `[600,1.10,10,10]` | stronger slow ramp |
| 10 | `[150,1.03,10,1]` | early small load |
| 11 | `[150,1.05,20,1]` | early moderate hold |
| 12 | `[150,1.08,10,10]` | early slow ramp |
| 13 | `[350,1.05,20,1]` | mid-cycle probe |
| 14 | `[450,1.05,20,10]` | mid-cycle slow ramp |
| 15 | `[750,1.08,10,1]` | very late probe |
| 16 | `[150,1.03,5,1];[600,1.08,10,1]` | early priming plus late response |
| 17 | `[600,1.05,5,1];[700,1.05,5,1]` | two separated short pulses |
| 18 | `[600,1.10,0,5]` | triangular pulse |

The sealed long-hold target `[300,1.10,100,1]` is not a candidate.

## Frozen screening rule

Simulate all 18 candidates in all 64 development cells with saturating SAC
and the same pre-paced cell state. Keep a candidate only if at least 61/64
cells pass QC. Among passing cells, compute response differences from the
same-cell unstretched baseline for V, Ca, and tension. Divide these channels
by fixed practical scales of 5 mV, 0.02 µM, and 2 kPa, sample every 25 ms,
and flatten the three-channel response for each cell.

For each candidate, form pairwise Euclidean distances among cells with valid
responses; compare candidates on the common valid cells by Spearman
correlation of these distance vectors. Candidate response energy is the
median across valid cells of the RMS normalized waveform difference. Discard
energy <0.1. Build a greedy library of at most 12 probes: first select the
highest-energy safe probe, then repeatedly choose the remaining probe with
lowest maximum correlation to the selected set, requiring every such
correlation to be <0.95. Ties use lower ID. No future target waveform or
parameter-recovery score enters selection.

## Gate

If fewer than 5 safe, nonredundant probes survive, stop before expensive
ensemble/OED development and report the action-space limitation. If at least
5 survive, proceed to target-specific plausible-twin disagreement on the
development cells, with fixed/random/parameter-information/goal-information
baselines and a **new** sealed confirmation cohort. Do not claim algorithmic
success from this library screen alone.

## Leave-one-cell-out acquisition pilot (frozen after library screen)

The library gate passed with IDs `12,6,11,8,5,7`. This next stage uses the
47 development cells whose saturating-SAC baseline/C1/C2/long-hold traces pass
existing QC. For each cell in turn, exclude it from the candidate population.
Build a **plausible analogue set**, not a calibrated posterior, from the 12
other cells with the smallest B+C1+C2 response distance. The calibration
distance is the mean squared normalized waveform difference over three arms:
baseline sampled every 25 ms with channel scales `[20 mV,.1 µM,2 kPa]`, and
C1/C2 response-minus-baseline in each onset-to-onset+200 ms window sampled
every 10 ms with scales `[5 mV,.02 µM,2 kPa]`. Analogue weights are Gaussian
in this distance with bandwidth equal to the 12th-neighbour distance, then
sum to one. The withheld cell's true parameters and future trace never enter
its candidate set or acquisition scores.

The future quantity is **target-minus-baseline V(t)** in 300–602 ms. A
probe's response is probe-minus-baseline V/Ca/T sampled every 25 ms. For each
candidate pair `j,k`, let `D_F` be future V MAE in mV and `D_P` be probe
response RMS using `[5 mV,.02 µM,2 kPa]` scales. Probe distinguishability is
`1-exp(-D_P^2/2)`. Candidate pair base weight is `q_j q_k`.

Six policy arms are compared. Adaptive arms choose among the six frozen
diverse probes; the earlier fixed C3 is retained as an external reference:

1. Fixed previous C3 reference, ID 4 (600 ms, λ=1.10, 20 ms).
2. Fixed highest-energy diverse probe, ID 12.
3. Balanced random benchmark, scored as the exact mean outcome across the
   six probes, avoiding random-seed noise.
4. Parameter-information: pair weight times mean squared prior-range-
   normalized 13-parameter distance.
5. Goal-information: pair weight times `D_F^2`.
6. Risk-targeted: pair weight times `max(D_F-5 mV,0)^2`; when every candidate
   pair is below 5 mV, fall back to goal-information.

For each selected probe, simulate its measurement by revealing only that
withheld cell's **development-library probe response**. Reweight the 12
analogues by a Gaussian likelihood in probe response distance with bandwidth
equal to the median pairwise probe distance among the analogues. Predict the
withheld future V trajectory from its observed baseline plus the weighted
analogue future deltas. This update is an engineering likelihood, not a
calibrated probabilistic posterior. Compare no-extra-probe, each policy,
mean/median/p90/CVaR90 V MAE, `>5 mV` counts, QC exclusions, and selection
frequency. This is an exploratory development pilot; any winning policy must
be frozen and rerun on a wholly new cohort before an algorithm claim.
