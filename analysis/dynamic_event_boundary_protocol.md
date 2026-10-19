# Mechanical long-hold event-boundary pilot (frozen before new scans)

Date: 2026-10-02. Exploratory within the existing saturating-SAC human
ToR-ORd/Land synthetic model. The three cells were selected because an earlier
tail audit implicated one parameter direction: cell 8 INaL, cell 43 IKr, and
cell 57 effective ionic feedback gain. These cells are **not** an independent
confirmation sample, and neither a sharp change nor a biological bifurcation
is assumed.

## Primary acute scan

For each cell, hold the pre-paced estimated initial state fixed. Change only
the implicated parameter from its inferred value to its known synthetic true
value at fractions `s=0:.125:1`. Run the same 300 ms onset, lambda=1.10,
100 ms hold trajectory with saturating SAC. The fixed-state scan measures acute
equation sensitivity; it is not a newly pre-paced cell population.

Record (a) upward -50 mV crossing in 300:602 ms; (b) time above -50 mV in
that window; (c) voltage peak and post-onset minimum; (d) window voltage MAE
against the existing true-cell waveform, clearly marked as diagnostic only;
(e) voltage derivative and effective direct stretch-current contribution near
the crossing or at 390 ms. Every simulation records numerical/QC failure.

## Pre-pacing and numerical follow-up

If a scan suggests a sudden event or a large plateau-duration change, select
the flanking parameter points without adapting the event definition. Re-pace
those modified parameters from the same estimated initial state for 60 beats,
record full-state relative last-beat change, then re-run the hold. A subset is
checked at stricter solver tolerance. A change that disappears after re-pacing
or solver refinement is a state/numerical effect, not an established dynamical
boundary. This pilot cannot prove a mathematical bifurcation or human-cell
phenotype; it can only prioritize further analysis.
