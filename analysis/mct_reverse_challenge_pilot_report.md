# Rabbit closed-loop mechanical challenge: first reverse-direction pilot

Date: 2026-09-29. This is a **rabbit MCT2024 in-silico development result**, separate from the one-way human ToR-ORd→Land cohort. The experimental Dryad example traces are not used to fit or validate the effects below.

## Question and design

Can a mechanical challenge reveal AP/Ca responses that are much less obvious under a weak load? We varied gel crosslinker concentration (`CL=0.02` and `0.10`) and the model's gel-stress→L-type Ca permeability feedback gain (`0, 0.5, 1, 1.5`). All runs used the author-supplied initial state, 1000-ms pacing, and 20 beats. The score is the final beat's Ca peak and time above −67 mV, **not APD90**. The raw model also provides voltage, calcium, shortening, gel stress, and the L-type Ca permeability multiplier.

The feedback-null control is informative: with gain 0, load still changes mechanics, but the final Ca peak changes by only 0.000015 µM and APD changes by 0 ms. At gain 0.5, 1, and 1.5, increasing load raises Ca peak by 0.108, 0.247, and 0.312 µM, respectively, and extends APD by 44, 87, and 105 ms. This is an intervention on the **model's feedback path**, not evidence that a specific biological cell has one of these gains. At high feedback, the loop also changes generated force and gel stress, so this is a full coupled response rather than a fixed-stress dose–response curve.

## Controlled virtual-variant cohort

We then crossed three feedback gains (`0.25, 0.75, 1.25`) with three basal L-type Ca permeability scales (`0.9, 1.0, 1.1`), creating nine controlled model variants. Each was simulated at both loads. The derivative MATLAB runner differs from the previously validated reference only by exposing these two multiplicative factors. At the reference setting (gain 1, permeability scale 1), the saved high-load voltage, Ca, time, and gel-stress arrays match the previous reference arrays exactly (maximum elementwise difference 0).

A useful pair emerged:

| Variant | Low-load APD (ms) | Low-load Ca peak (µM) | High-load APD (ms) | High-load Ca peak (µM) |
|---|---:|---:|---:|---:|
| Gain 0.25, Ca permeability ×1.1 | 277 | 0.462 | 306 | 0.542 |
| Gain 1.25, Ca permeability ×1.0 | 271 | 0.445 | 368 | 0.730 |

The low-load pair differs by 6 ms and 0.017 µM, while the high-load pair differs by 62 ms and 0.189 µM. We selected this pair *after inspecting all nine variants*, so this is a demonstrative example rather than a frequency estimate. The two parameters can partly offset each other at low load but respond differently once mechanical feedback is engaged.

Because 20 beats still showed some drift, we reran this pair for 40 beats. Their low-load differences were **4 ms and 0.015 µM**; high-load differences were **56 ms and 0.182 µM**. Both high-load APD traces changed by −3 ms from beat 37 to beat 40. This supports the existence of the challenge-unmasked difference within this model, although it is not proof of exact steady state.

As a rough *development diagnostic*, leave-one-variant-out ridge regression using low-load APD and Ca peak to predict high-load values had MAE 28.1 ms and 0.081 µM across the nine variants. Predicting no load effect gave 56.9 ms and 0.159 µM. Nine hand-picked grid points are much too few for a robust ML comparison, and these numbers are **not** held-out population performance.

## Biophysical interpretation and boundary

Within MCT2024, increased gel loading raises gel stress. The feedback term changes L-type Ca permeability; Ca cycling and AP repolarization then change, which also alters contraction and stress. The zero-gain control locates the AP/Ca load effect in that implemented feedback path. The gain scan tests sensitivity to the *strength of that implemented path*; it does not identify a molecular gain in rabbit cells or provide a coefficient transferable to human ToR-ORd/Land.

`CL` is a gel crosslinker setting, not an externally measured force waveform. There are only two mechanical conditions in this pilot, no independent biological cells, no alternans assessment, and no real-data predictive validation. The human platform remains AP→Ca→tension; it has not acquired reverse mechanical feedback from this exercise.

## Decision

The reverse-direction question is computationally viable: a mechanical challenge can expose electrical/Ca differences obscured under a weak load in the native closed-loop rabbit model. The next benchmark should vary more than two physiological parameters, use an explicit convergence rule, include a third load for calibration versus blind challenge, and keep all model variants from the same parameter draw together across train/test splits. It should compare baseline-only, low+intermediate-load, and mechanistic-twin predictions of a predeclared high-load AP/Ca endpoint; alternans needs beat-resolved analysis and must be scored separately.

## Reproduction and artifacts

- Generate the parameterized runner: `analysis/prepare_mct_reverse_variant.py`.
- Run the controlled cohort: `analysis/mct_feedback_audit/run_mct_reverse_cohort.m`.
- Run the 40-beat pair check: `analysis/mct_feedback_audit/run_mct_reverse_pair_40.m`.
- Score: `analysis/mct_reverse_challenge_pilot.py` and `analysis/score_mct_reverse_cohort.py`.
- Numeric data and figures: `analysis/mct_feedback_audit/reverse_challenge_pilot.json`, `reverse_cohort_score.json`, `reverse_challenge_pilot.png`, and `reverse_cohort_response.png`.
