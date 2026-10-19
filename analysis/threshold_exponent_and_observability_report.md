# Threshold-aware energetic reanalysis and digital-twin observability

2026-09-28. Analysis unit is the independent cell. Scripts and machine-readable results are `threshold_aware_exponent_audit.py` / `.csv` / `.json`, `mechanotransduction_observability_framework.py` / `.json`, and `observability_noise_demo.py` / `.json`. All outputs can be regenerated using `.venv_tcn/Scripts/python.exe`.

## 1. Why audit the exponent?

[Young et al., PNAS 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974467/) report work-current power coefficients of Piezo1 6.2±0.4 (n=39 cells), Piezo2 2.6±0.2 (n=47), TRAAK 1.9±0.2 (n=38), and TREK1 1.4±0.1 (n=60). Their stated procedure removes pre-threshold points, log-transforms work and absolute current, and uses a linear fit. Thus the reported coefficient is the apparent log-log slope **after truncation at an estimated response threshold**.

If a candidate underlying law were (I=A(W-W_0)^\beta), then its instantaneous apparent unshifted slope would be (\alpha(W)=\beta W/(W-W_0)). This algebra raises a testable concern; it does **not** establish that the shifted law is true.

## 2. Cohort and reproducibility

I used the supplied `bychannel_summary.csv` selected sweep and matching `bychannel_approaches.csv`, requiring the metadata `responds=1`, at least 50 positive post-threshold current samples and at least 0.05 log10 work span. This yields 19 Piezo1, 13 Piezo2, 23 TRAAK, and 32 TREK1 cells. Eleven cap-chimera (`chimA`) cells were processed independently. The numbers are smaller than the paper's full reported n, so this is a **public selected-sweep subset audit**, not an exact numerical replication of all figures. There are 12 Piezo1 dates, 4 Piezo2 dates, 8 TRAAK dates, 10 TREK1 dates, and 2 chimera dates; channel and date cannot be cleanly separated.

For each cell, I refit the original form, \(\log I=c+\alpha\log W\), and a fixed-threshold shifted form, \(\log I=c+\beta\log(W-W_{th})\), on **identical samples**. I did not estimate a new threshold from the current. I also computed 0.10-decade equal-width early and late local slopes, with one contribution per cell. Confidence intervals resample cells and, separately, acquisition dates within channel; neither treats adjacent timepoints as independent.

| Channel | Cells | Median original α | Median shifted β | Median log10 work span | Median R² original / shifted |
|---|---:|---:|---:|---:|---:|
| Piezo1 | 19 | 4.98 | 0.58 | 0.20 | 0.907 / 0.650 |
| Piezo2 | 13 | 2.97 | 0.58 | 0.43 | 0.981 / 0.703 |
| TRAAK | 23 | 1.64 | 0.32 | 0.40 | 0.880 / 0.576 |
| TREK1 | 32 | 1.36 | 0.35 | 0.67 | 0.930 / 0.722 |
| Piezo1-cap2 chimA | 11 | 4.27 | 0.63 | 0.32 | 0.983 / 0.562 |

The original α ordering is present in this subset. Piezo1 minus Piezo2 median α is 2.01, cell-bootstrap 95% interval 0.49–3.17; date-cluster interval 0.53–4.23. The shifted β difference is -0.005, cell-bootstrap interval -0.33–0.14, date-cluster interval -0.26–0.12. However, the shifted law has higher within-trace log R² in only **6/87 WT cells**. Similar β values are therefore a **diagnostic**, not evidence that a shared shifted-power mechanism fits these traces.

Across all WT cells, span versus α Spearman ρ=-0.61. It is also negative within each channel (P1 -0.53, P2 -0.73, TRAAK -0.39, TREK1 -0.38). This association is partly mathematical: a small available log-work span and a threshold close to the observed work range amplify the unshifted log slope. It does not establish a biological cause.

## 3. Shared-exponent null and range matching

I used every cell's observed work trajectory and reported threshold, generated \(I=A_i[W-W_{th,i}]_+^{\beta}\) with one shared \(\beta=0.447\) (median shifted slope across the analyzed cells), scaled each trace to its observed post-threshold maximum, and refit the original unshifted exponent. Adding Gaussian noise at the supplied cell-specific baseline SD and doubling that noise barely changed the medians. At baseline noise, 300 simulated replicates gave a P1–P2 apparent α difference of **1.50** (2.5–97.5% simulation interval 1.42–1.59), versus **2.01** observed. This is a conditional simulation interval for the chosen trajectories and thresholds, **not** a sampling confidence interval or a claim that 75% of a biological effect is explained.

The same null placed TRAAK slightly *above* Piezo2 (median α 1.66 vs 1.59) and reproduced the complete four-channel ordering in **0/300** baseline-noise replicates. Therefore threshold/work-range geometry is sufficient to create a large P1–P2 contrast but is insufficient here to explain the full observed order. Other shape differences, threshold-estimation bias, current kinetics during loading, and channel effects remain possible.

Equal-width 0.10-decade local fits sharpen that boundary. Early-window P1–P2 median-slope difference is 0.06 (cell-bootstrap interval -1.77–3.47); late-window difference is 3.41 (2.24–5.38). The result depends strongly on where the work window is placed, so one truncated global α should not be read as an invariant channel property. The late difference also argues against declaring all P1/P2 shape differences eliminated by range matching. The chimera has intermediate original α but only two dates and no matched same-date WT comparison; it does not resolve the question.

**Go/no-go:** The strong claim “Young's four-channel ordering is entirely a threshold artifact” is **no-go**. The narrower, useful claim “the published unshifted α is substantially sensitive to threshold and dynamic range, so its magnitude cannot uniquely encode a channel-specific power-law mechanism” is supported in this subset. A corrective paper attacking the original channel conclusion would require all published cells and stronger model comparison; the current selected-sweep subset does not justify that claim.

## 4. Exact observability result

For a static tension sigmoid,

\[
I(F)=G\,\sigma\!\left(\frac{T_0+\lambda F-T_{50}}{k}\right)
=G\,\sigma\!\left(\frac{F-F_{50}}{s_F}\right),\quad
F_{50}=\frac{T_{50}-T_0}{\lambda},\quad s_F=\frac{k}{\lambda}.
\]

The transformation \((T_0,\lambda,T_{50},k,G)\mapsto(aT_0+b,a\lambda,aT_{50}+b,ak,G)\), \(a>0\), leaves **every force-current prediction exactly unchanged**, for any number of noiseless sweeps. Symbolic Jacobian ranks of the observable parameter combinations are 3/5 for a rich same-cell \(F,I\) curve, 4/5 after calibrated same-cell \(\Delta T\), and 5/5 after calibrated same-cell absolute tension \(T\). The latter ranks are ideal structural upper bounds, not promises under Duke's limited stimuli.

| Same-cell observations | Candidate identifiable quantities with rich input | Exact ambiguity left |
|---|---|---|
| \(F,I\) | \(F_{50},s_F,G\) | Tension scale and zero |
| \(F,\Delta T,I\) | Above plus \(\lambda\), hence \(k\) and \(T_{50}-T_0\) | Absolute tension zero |
| \(F,T_{abs},I\) | Above plus \(T_0\), hence \(T_{50}\) | Electrical product \(G=N\gamma(V-E)\) |
| Plus multiple holding voltages | \(E\) and \(N\gamma\), if voltage-independent gating/conductance assumptions hold | Number \(N\) versus single-channel conductance \(\gamma\) |
| Plus channel count or calibrated single-channel current | Can separate the remaining product under those assumptions | Spatial heterogeneity/model misspecification |

This extends beyond the sigmoid: a hard threshold gate has an affine-tension symmetry and reveals only an effective force threshold and gain; a Hill gate referenced to zero tension retains a *scale* symmetry but generally not the sigmoid's offset symmetry; a linear latent-tension relaxation ODE retains the affine symmetry exactly. Appending an adaptation state does not itself anchor absolute tension. The symbolic checks are in `mechanotransduction_observability_framework.py`.

A deliberately synthetic 30-force-point example (current noise 5 pA; tension noise 0.03 mN/m) makes the distinction visible. The output loss is flat to numerical precision across both tension scale and offset with \(F,I\); calibrated \(\Delta T\) removes the scale flat direction but leaves offset flat; absolute tension removes both. The input grid is deliberately broad, so this is an **identifiability demonstration**, not a prediction of achievable precision in the existing datasets. Results and a figure are in `observability_noise_demo.json` and `.png`.

Crucial transport caveat: ETH tension, eLife patch gating, and Duke current are from different cells, geometries and protocols. They are independent physical constraints or priors. They cannot be stacked as though they were same-cell \(F,T,I\) observations to eliminate Duke's gauge symmetry. Exact prior knowledge of \(T_{50},k\) *in Duke's cell context* could in principle anchor the sigmoid; one-shot calibration and narrow force ranges still pose a practical inference problem.

## 5. Decision

The exponent audit produced a defensible **measurement-geometry caution**, but not a decisive revision of the original biological ranking. The most solid pure-computation result now is the **model-family-specific observability map**: it states exactly which biological parameters are only equivalent descriptions of one force-current curve and which same-cell measurements would separate them. The next paper-quality step would be a broader theorem/benchmark with explicit protocol-dependent practical uncertainty and at least one external cohort where same-cell multimodal observations actually remove a predicted ambiguity. Without that external validation, the present symbolic map is a rigorous methods foundation, not yet a high-impact paper claim.
