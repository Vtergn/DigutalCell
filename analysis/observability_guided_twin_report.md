# Ridge-seeded mechanistic refinement: blind-load confirmation

## Question and pre-existing benchmark

The earlier 5,000-cell ToR-ORd→Land benchmark showed that two observed mechanical loads improve inverse observability, yet a one-pass neural posterior propagated through the true equations predicted λ=1.125 poorly (7.45 kPa waveform MAE on 16 held-out cells). A direct neural predictor and simple two-load linear extrapolation reached 3.21 and 3.61 kPa on those same cells. The new analysis tests whether the inverse step, rather than the forward equations, is the main bottleneck.

All cells remain separated by the original frozen train/validation/test split. The model is an adult ventricular **one-way** excitation→calcium→contraction simulator. No MCT feedback, experimental patient data, simulator mismatch, or observation noise is present.

## Local observability diagnostic

Central finite differences of the *true* ToR-ORd/Land equations were computed for all 12 prior-scaled parameters in four diverse **training** cells. Each perturbation was 1% of the parameter's prior width, with the same parameter vector evaluated at λ=1.0, 1.1, and the sealed target 1.125. AP, Ca, and tension derivatives were scaled by 100 mV, 0.5 µM, and 50 kPa, respectively, and by the square root of each trace's sample count. With a relative singular-value cutoff of 5%, the local subspace spanned by AP+Ca captured 34.6% of the *target derivative energy*; AP+Ca+T(1.0) captured 85.5%; adding T(1.1) captured 99.7%. These percentages are exploratory local geometry under a declared scaling/cutoff, **not** calibrated Fisher information, global identifiability, or proof that all cell parameters are recoverable. In all four cells, the numerical derivatives with respect to β0 and β1 were exactly zero at λ=1.0, as required by the Land equations, and nonzero after adding λ=1.1.

## Algorithm: ridge start, five-parameter true-Land correction

1. Fit an ordinary waveform-to-12-parameter ridge model on training cells, using AP, Ca, T(1.0), and T(1.1). For final test cells, refit ridge on training+validation cells, following the earlier benchmark protocol. Clip initial parameters to the declared prior bounds.
2. Condition the **actual Land ODE** on the observed baseline Ca waveform. This is legitimate in this one-way simulator because Ca does not change with stretch. A true-parameter Land-only reproduction of a stored cell's T(1.0) trace differed by at most 0.037 kPa due to numerical interpolation.
3. Keep three weakly constrained Land parameters at their ridge start; optimize only Tref, ca50, nperm, β0, and β1. Minimize waveform residuals at observed λ=1.0 and 1.1, plus an L2 penalty around the ridge start (γ=0.1 on prior-unit parameters). The solver is MATLAB `lsqnonlin` with at most five iterations and 45 evaluations. No target-load waveform enters this objective.
4. Run the calibrated Land parameters at λ=1.125 through the actual ODE. This is a **point-estimate mechanistic twin**, not yet a parameter posterior or a full bidirectional electromechanical simulator.

The four validation cells improved from 8.27 to 1.58 kPa mean target waveform MAE with this fixed setting. On the original exploratory 16 test cells, mean MAE was 0.91 kPa versus 3.21 direct neural and 3.61 linear extrapolation; 14/16 improved relative to the unrefined Ridge/Land start. This motivated one larger, sealed confirmation rather than further tuning.

## Fresh 128-cell target-blind confirmation

The confirmation cells were deterministically sampled from the original test split **excluding the earlier 16 cells**. The MATLAB input file contains only cell IDs, observed Ca, observed T(1.0)/T(1.1), ridge starts, and prior bounds. It contains **no λ=1.125 curve or true parameter vector**. γ=0.1 and the five fitted coordinates were fixed before scoring this sample. The target was read only after all 128 fits had finished.

| Method | λ=1.125 waveform MAE, mean (kPa) | Median (kPa) | Peak-amplitude MAE, mean (kPa) |
|---|---:|---:|---:|
| Ridge → Land, before correction | 2.665 | 2.379 | 3.890 |
| **Ridge → true-Land correction** | **1.193** | **0.968** | **2.117** |
| Direct neural waveform predictor | 2.529 | 1.759 | 3.526 |
| Two-load linear extrapolation | 3.074 | 2.509 | 4.246 |

The refined twin improved the waveform MAE over the direct network for **99/128** cells and over linear extrapolation for **117/128**. The paired mean advantage is 1.336 kPa over direct neural [0.996, 1.698] and 1.881 kPa over linear extrapolation [1.564, 2.218]; intervals are 5,000 cell bootstrap resamples, exploratory rather than independent-replication confidence. The calibration-load MAE decreased from 1.492 to 0.651 kPa. Seven fits stopped at the iteration/evaluation limit (exit flag 0); they remain included. Median/maximum function calls were 24/42 per cell, so this is substantially more expensive than a direct neural inference.

Notably, mechanical parameter MAE normalized to prior width changed only from about 0.193 at the Ridge start to 0.190 after correction. **Better unseen-load prediction did not require recovering every original Land parameter.** The result supports task-relevant identifiable combinations, while also limiting claims about molecular parameter identification.

## Interpretation and limits

This is a positive algorithm result **within an exact, noise-free simulator family**: a small, observation-constrained mechanistic correction outperforms a trained direct predictor on a nearby unseen stretch. It does not establish superiority on real cardiomyocytes, broad stretch extrapolation, drug perturbations, model mismatch, noisy recordings, or calibrated uncertainty. The target λ=1.125 is 0.025 beyond the maximum observed calibration stretch and 0.05 beyond the highest supervised direct-network target; this is a narrow extrapolation test. The method uses observed Ca, which is load-invariant by construction in this one-way benchmark; a closed-loop human model would require a different inference/forward protocol. The next substantive gates are measurement-noise/model-mismatch robustness, multiple target loads, and posterior calibration around the corrected identifiable directions. Adding a neural forward residual to this exact-simulator benchmark would confound this result.

## Reproduction and audit files

- `export_jacobian_bases.py`, `run_local_jacobian.m`, `score_local_jacobian.py`: local true-equation Jacobians and SVD summary (`virtual_cardiomyocyte_5000/local_jacobian_audit.json`).
- `export_ridge_twin.py`, `run_posterior_draws.m`, `score_ridge_twin.py`: same-16-cell ridge point twin versus existing methods (`virtual_cardiomyocyte_5000/ridge_true_forward_16_score.json`).
- `land_tension_given_ca.m`: true Land forward conditioned on the supplied Ca trace.
- `prepare_land_refinement.py`, `refine_land_one_cell.m`, `refine_land_batch.m`, `score_land_refinement.py`: four validation and original 16-cell exploratory test (`virtual_cardiomyocyte_5000/land_refinement_score.json`).
- `prepare_land_blind_confirm.py`, `score_land_blind_confirm.py`: sealed new 128-cell cohort, target-blind input, scoring, and paired comparison (`virtual_cardiomyocyte_5000/land_blind_128_selection.json`, `land_blind_128_score.json`, `land_blind_128_comparison.png`).
