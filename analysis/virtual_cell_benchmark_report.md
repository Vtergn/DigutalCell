# Paired virtual cardiomyocytes: cohort and inference pilot

## Simulator and benchmark definition

The synthetic cell has 12 parameters: four ToR-ORd multipliers (`ICaL`, `Jup`, `INaCa`, `IKr`) and eight Land parameters (`Tref`, `ca50`, `TRPN_n`, `nperm`, `beta0`, `beta1`, `phi`, `perm50`). Parameter bounds, random seed, cell split, and condition list are saved with every cohort. Each cell has one AP and Ca trace, then active/passive tension at seven isometric stretches: 0.975, 1.000, 1.025, 1.050, 1.075, 1.100, 1.125. The two endpoints outside [1.0,1.1] are reserved for load extrapolation tests.

The ToR-ORd→Land link is **one-way**, and this benchmark represents adult ventricular cardiomyocytes, not hiPSC-CMs. `Land2016_variable.m` is a locally parameterized adaptation of the downloaded Land source. With all parameters at the Land defaults, its derivative and tension matched the unmodified source exactly in a numerical check. `torord_land_given_theta.m` independently regenerated a stored cohort waveform at λ=1.125 to within `1.90×10⁻⁶ kPa` (single-precision storage error), establishing that posterior samples can return to the true forward equations.

## Pilot quality gate

The revised 256-cell cohort passed at **256/256**. Per-cell electrical pacing was extended from 12 to 20 beats when needed; the largest accepted Ca cycle difference was 0.99% of peak. Land was conditioned for 5 cycles, or 7 if needed; the largest accepted tension cycle difference was 0.85% of peak. Failure counts, if any, remain visible in individual shard files and merger metadata.

This pilot's active-tension peak distribution is still lower than the published Strocchi emulator-training population: at λ=1.0, the medians are **17.7 vs 41.5 kPa**; at λ=1.1, **50.8 vs 60.4 kPa**. We corrected one clear prior mismatch (`nperm`/`nTm` range) by looking at published parameter ranges, not by fitting to target outputs. We have **not** tuned the synthetic population to reproduce Strocchi feature histograms. The benchmark prior is therefore explicitly synthetic and somewhat broad toward weak contraction.

## First observability result (cell-held-out ridge diagnostic)

On the frozen 179/38/39 train/validation/test split, mechanical-parameter MAE divided by prior width was:

| Observation | Frozen-test mechanical-parameter MAE |
|---|---:|
| AP | 0.246 |
| AP + Ca | 0.242 |
| AP + Ca + T(1.0) | 0.224 |
| AP + Ca + T(1.0) + T(1.1) | **0.205** |

Paired bootstrap reduction from AP+Ca is 0.0186 [0.0088, 0.0285] after one load, and 0.0378 [0.0258, 0.0505] after two loads (5,000 cell resamples; exploratory confidence intervals). The 5-fold cross-validation pattern is similar. The largest clear parameter-specific gain is `beta1`: its 5-fold held-out R² changes from approximately −0.03 with AP+Ca, to −0.07 with T(1.0), to **0.53** after adding T(1.1). Several other mechanical parameters remain weakly identified; do not interpret aggregate improvement as recovery of every parameter.

There is also an exact structural result from the Land equations. At λ=1, the length factor is `Lfac = 1 + beta0 × (λ−1) = 1`, while the length-dependent calcium threshold shift is `beta1 × min(0.2, λ−1) = 0`. Consequently, **β0 and β1 do not affect AP, Ca, or T(1.0)** in this one-way benchmark, no matter how many perfect repeated traces are collected at λ=1. At λ=1.1 they enter the contraction equations. This is an explicit example of how load diversity can create parameter observability; finite data may still leave practical ambiguity (especially β0).

## Direct prediction baseline and its implication

A small load-conditioned convolutional predictor was trained on the 256-cell pilot. Its training target loads exclude λ=1.125. With T(1.0) and T(1.1) calibration, its frozen-test λ=1.125 waveform MAE is **7.31 kPa**. A simple linear extrapolation of the two measured tension traces gives **3.36 kPa** on the same cells. At λ=1.05, the network gives 1.83 kPa and linear interpolation 1.97 kPa. This is preliminary and size-limited, but it fixes an essential comparison: the mechanistic posterior twin must beat or meaningfully complement both a learned direct predictor **and** simple load interpolation/extrapolation.

The `LoadConditionedPosterior` architecture is implemented and has passed finite-density, sampling, missing-modality, and load-order invariance smoke checks. After fitting train-only waveform normalization, a 256-cell training pilot improves electrical median parameter error from 0.201 (AP) to 0.125 (AP+Ca), but its mechanical median error stays near 0.24 even with two loads. This is **not yet a successful posterior individualizer**: the much simpler ridge diagnostic reaches 0.212 for mechanical parameters on the same cohort. Larger-cohort training and posterior calibration remain necessary; the pilot does not establish the proposed AI method.

## Reproduction files

- `torord_land_cohort.m`: deterministic parameter sampling, per-cell forward simulation, quality checks, frozen splits, checkpointing.
- `Land2016_variable.m`: eight mechanical parameters exposed without changing default behavior.
- `torord_land_given_theta.m`: true forward propagation for arbitrary explicit θ and supported λ.
- `merge_virtual_cell_shards.py`: combines completed independent shards while recording failure counts.
- `virtual_cell_pilot_audit.py`: source-population comparison and held-out-cell ridge diagnostic.
- `load_conditioned_posterior.py`, `train_load_posterior.py`: permutation-invariant mechanical observation encoder and mixture posterior training/evaluation.
- `train_direct_load_predictor.py`: load-conditioned direct waveform and static load baselines.
- `virtual_cardiomyocyte_256/`: validated pilot data, parameter table, frozen splits, audit, figures, direct models.

## Claim boundary

The pilot establishes a working same-parameter AP/Ca/T simulator and an initial observation-dependent recovery pattern. It does **not** yet establish calibrated parameter posteriors, true-forward posterior prediction, a benefit over two-load linear extrapolation, a physiologically representative human-cell population, or mechanoelectric feedback. Those remain separate tests on the large cohort.

## Full 5,000-cell benchmark (September 2026)

Eight independently generated 625-cell shards were merged after all **5,000/5,000** cells passed the electrical and mechanical checks. Frozen splits contain 3,496 training, 744 validation, and 760 test cells. All figures below use these frozen splits. The full synthetic prior remains relatively weakly contractile compared with the Strocchi reference population: median active-tension peaks are 18.2 kPa at λ=1.0 and 50.8 kPa at λ=1.1. The population is a controlled benchmark, not a validated human physiological distribution.

The simple waveform-to-parameter ridge diagnostic replicated and strengthened the pilot observability result:

| Available baseline modalities | Test mechanical-parameter MAE / prior width | Test β1 R² |
|---|---:|---:|
| AP + Ca | 0.251 | −0.016 |
| AP + Ca + T(1.0) | 0.221 | −0.022 |
| AP + Ca + T(1.0) + T(1.1) | **0.198** | **0.561** |

The paired mechanical-MAE reduction relative to AP+Ca is 0.0301 [0.0281, 0.0325] with one load and 0.0531 [0.0504, 0.0560] with two loads (3,000 bootstrap resamples over 760 held-out cells). β0 remains weakly recovered even with two loads (R² 0.025). At λ=1.0, β0 and β1 are exactly absent from the equations; one load cannot identify them even with noiseless traces.

The set-conditioned three-component posterior network improves median electrical error from 0.144 with AP to 0.0051 with AP+Ca. Its mechanical median error changes only from 0.253 with AP+Ca to 0.234 with T(1.0) and 0.232 with T(1.0)+T(1.1), substantially worse than the ridge diagnostic's 0.198. Mechanical marginal 90% intervals cover 90.2% of parameter values with two loads, but this aggregate calibration does not rescue β1 recovery: its median error stays at 0.244 versus ridge 0.153. Thus the proposed posterior architecture has **not** yet learned the available load-specific information effectively.

For the sealed outside-range load λ=1.125, the direct waveform network trained on target loads at or below λ=1.1 achieves 4.73 kPa MAE with T(1.0) calibration, versus 19.68 kPa for a constant-trace baseline. With T(1.0)+T(1.1), the direct network achieves **2.50 kPa** versus **3.09 kPa** for linear extrapolation over all 760 test cells. This is a modest but genuine out-of-range advantage for the learned direct predictor; its λ=1.05 in-range MAE is 0.152 kPa versus 1.809 kPa linear interpolation.

The posterior was also propagated through the *actual* ToR-ORd and Land ODEs, not a learned surrogate: 16 randomly selected held-out cells, eight posterior draws per cell, two calibration modes, all 256 ODE simulations successful. On those same 16 cells at λ=1.125, one-load posterior median waveform MAE is 8.99 kPa versus 6.38 direct neural and 21.44 constant-trace. Two-load posterior median MAE is **7.45 kPa** versus **3.21 direct neural** and **3.61 linear extrapolation**. Its eight-draw pointwise 90% interval covers 91.4% of target time points, but has a very wide average width of 37.6 kPa, and the sample is too small to claim calibrated uncertainty. The two-load twin improves its own one-load result, yet fails the pre-specified competitiveness gate. More particles or another surrogate are not indicated; improving mechanical posterior inference is the bottleneck.

Reproducible result files are `virtual_cardiomyocyte_5000/full_cohort_observability.json`, `virtual_cardiomyocyte_5000/merged_cohort_posterior/results.json`, `virtual_cardiomyocyte_5000/direct_1load/results.json`, `virtual_cardiomyocyte_5000/direct_2load/results.json`, and `virtual_cardiomyocyte_5000/posterior_true_forward_16x8_score.json`. The out-of-range load was withheld as a *target* from network training; the same simulator family and prior generated both train and test cells, so this is out-of-condition, not out-of-model generalization.

## Newly downloaded mechanical-feedback resources

The Dryad B8T935 `README.pdf` identifies **rabbit left-ventricular myocytes**. Its `Experimental_Data.zip` contains 4-ms-sampled, 6.288-s example traces for Fura-2 Ca ratio and sarcomere length under load-free and in-gel conditions, each with raw and smoothed versions. The gel example has a higher Fura-2 peak (~3.07 versus ~2.03), but different baseline sarcomere lengths and only one trace per signal/condition are present. No cell IDs, replicate table, or explicit same-cell pairing are supplied. Consequently these files support a descriptive direction check, not paired effect estimation, posterior training, or direct calibration of an absolute Ca concentration.

The downloaded MCT2024 package is a **different rabbit Shannon–Bers/Negroni closed-loop model**, not a ToR-ORd/Land add-on. Its README prescribes a large pacing-cycle-length sweep and refers to an initial-state file `yfinal_3000.mat`, while the archive contains `yfinal_2000.mat`. A generated local reference runner retains the original ODEs, uses the available initial state, and fixes PCL=1000 ms for 20 beats. Its only equation-level alteration for the ablation is an explicit feedback gain: gain=1 reproduces the original gel-stress-dependent L-type Ca permeability; gain=0 holds that multiplier at one. The gain=1 runner's first eight beats reproduce a separate unmodified-equation pilot's Ca trajectory to a maximum absolute difference of 1.03×10⁻⁹ µM.

Comparing gel crosslinker concentration 0.02 versus 0.10 in this **native rabbit model** gives last-beat Ca peaks 0.426→0.673 µM and AP duration above −67 mV 264→351 ms with MCT on. With MCT off, Ca peaks are 0.361→0.361 µM and AP duration 242→242 ms. The difference between MCT-on and MCT-off is about 4.9 times larger under the higher-load condition for both Ca peak and AP duration. This is a concrete single-model illustration of a mechanical challenge revealing feedback that is less prominent at low load; it is not population-level evidence of a hidden phenotype. Gel stress and shortening still change when feedback is off, as mechanics should. Last-beat Ca peak drift is below 0.12% relative to the previous beat in every condition. This is a successful *model ablation*, not new experimental evidence and not validation of these parameters in the ToR-ORd/Land human-cell benchmark.

The experiment audit, generated native-model runner, raw ablation traces, quantitative score, and plots are under `analysis/mct_feedback_audit/`. The immediate modeling implication is to keep two branches: (1) the existing one-way human electromechanical benchmark and (2) a rabbit closed-loop reference for defining feedback observables and checking qualitative direction. A future unified human feedback model requires an explicit stress observable and independently justified stress-to-current map; copying the rabbit permeability coefficients into ToR-ORd/Land would not be identifiable or validated with these files.

## Subsequent inverse-algorithm rescue

The separate `observability_guided_twin_report.md` tests a fixed Ridge-initialized, five-parameter true-Land calibration on the same frozen cohort. In a fresh target-blind sample of 128 test cells, λ=1.125 mean waveform MAE is **1.193 kPa** versus 2.529 direct neural and 3.074 linear extrapolation. This supersedes the earlier conclusion that the mechanistic route cannot predict competitively *in this exact synthetic benchmark*. It does **not** rescue the original one-pass posterior network, establish calibrated uncertainty, or validate a human closed-loop model.
