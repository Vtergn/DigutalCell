# Sealed test of a model-based long-hold risk warning

Frozen 2026-10-02 before creating the new-seed confirmation cohort or
examining its target responses.

## Hypothesis

The local long-hold waveform sensitivity seen in cells 8 and 43 might be
useful as a **prospective error warning** for a newly individualized cell.
The score must use only baseline/C1/C2 observations and its inferred native
model. It may simulate the prespecified long-hold protocol under perturbations
of inferred parameters, but may not use true parameters or target waveforms.

## Cohort and prediction

Create 64 new 13-parameter Latin-hypercube synthetic cells with seed
`20261021`, distinct from all previous cohorts. Use the existing development
47-cell B+C1+C2 PLS20 hard-clipped inverse without refitting, and the same
native saturating-SAC forward model. Truth QC, inverse pre-pacing QC, and the
300–602 ms target V-MAE score follow the previous C3 analysis. Native solver
failures count separately; risk discrimination is only evaluated among common
QC cells and must not hide exclusions.

## Precommitted local risk score

At the inferred, re-paced cell state, simulate the 300-ms-onset 100-ms-hold
protocol. Change each of three previously implicated parameters separately:
IKr (index 4), INaL (index 6), and effective ionic feedback gain (index 7).
Use +/-2.5% of that parameter's full prior range, clipped to prior bounds.
Hold the same initial state for these acute finite differences. For each
perturbed waveform, compute its mean absolute V difference from the central
waveform over 300–602 ms. The score is the maximum of the six differences.
If a perturbation fails QC, record failure and assign infinite risk for
ranking; report the failure rate. This is an *uncertainty sensitivity proxy*,
not a posterior or calibrated error probability.

Compare to three simple, also target-blind scores: (1) count of inferred
parameters exactly on bounds, (2) long-hold predicted voltage duration above
-50 mV, and (3) maximum B/C1/C2 calibration reconstruction residual if
available; if not available, explicitly mark that baseline untested rather
than replace it after seeing results. Primary discrimination is AUROC for
`V MAE > 5 mV`; because events may be rare, also report average precision,
top-decile recall, Spearman with continuous V MAE, and per-cell scatter.
Bootstrap cell-level intervals. No score threshold is tuned on confirmation.

## Decision

If the sensitivity score does not beat the simple bounds/predicted-duration
scores or misses severe tail cells, do not claim a predictive event margin.
Even a positive same-model result remains a synthetic warning-signal pilot,
not a neural representation result or cross-model validation.

## Replication added before generating its cells

The first 64-cell cohort had only two `>5 mV` events, although both ranked
highest by the frozen score. To test whether that ordering survives more
events, generate a **second independent 128-cell Latin-hypercube cohort** with
seed `20261023`. Reuse the same 47 development cells, PLS20 inverse, native
forward model, +/-2.5% perturbations, three prespecified parameters, QC,
baselines, and scoring code without changing any threshold or weights. Report
both cohorts separately and pooled only as secondary context. The second
cohort is not used to select or calibrate the score.
