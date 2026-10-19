# Predictive-disagreement probe design: development feasibility result

Date: 2026-10-03. This is a **development-only, same-model synthetic pilot**, not an independent validation of an adaptive design algorithm. The candidate probes and evaluation rules were frozen in [predictive_disagreement_design_protocol.md](predictive_disagreement_design_protocol.md) before the corresponding simulations and acquisition pilot.

## What was tested

The 64-cell development population was simulated under 18 mechanical probes. All 18 passed the existing finite-value and broad V/Ca/tension range checks in all 64 cells. Eleven cleared the prespecified response-energy threshold. The prespecified response-diversity filter retained six probes: IDs **12, 6, 11, 8, 5, 7**. These vary onset, amplitude, hold time, and ramp time. This passes the **action-space feasibility gate**, but simulated QC does not establish safety in a real cell experiment.

As a numerical cross-check, regenerated probe ID 4 matched the previous C3 reference waveform exactly in the first eight cells (`max absolute difference = 0` at stored precision).

For the next gate, 47 cells had valid baseline, C1, C2, and sealed long-hold traces. Each cell was withheld from a 12-cell nearest-analogue ensemble built only from the other development cells' baseline/C1/C2 observations. The analogue ensemble is **not** a calibrated posterior. Acquisition scores used the analogues' long-hold predictions but never the withheld cell's long-hold result. After selecting a probe, its observed response in the withheld cell was revealed and used for analogue reweighting. All methods had one additional probe at the same budget.

## Long-hold V waveform error

| Method | Mean MAE (mV) | Median | p90 | CVaR90 | Cells >5 mV |
|---|---:|---:|---:|---:|---:|
| No extra probe | 12.04 | 7.08 | 31.46 | 54.00 | 32/47 |
| Earlier fixed C3, ID 4 | **9.16** | **3.49** | **27.97** | **45.94** | **22/47** |
| Fixed highest-energy diverse probe, ID 12 | 11.25 | 6.66 | 30.29 | 53.62 | 27/47 |
| Uniform random across six, exact policy mixture | 10.29 | 4.85 | 41.62 | 50.29 | 23/47 expected |
| Parameter-information selection | 11.28 | 4.73 | 41.62 | 51.38 | 23/47 |
| Future-goal information selection | 11.09 | 4.84 | 41.52 | 49.92 | 23/47 |
| Future-dangerous-pair selection | 11.09 | 4.84 | 41.52 | 49.92 | 23/47 |

The risk-targeted score selected the **same probe as the goal-information score for all 47 cells**, including nine fallback cells. It selected ID 12 in 28 cells, ID 5 in 16, and ID 6 in three. Its error was lower than the older fixed C3 in 14 cells and higher in 33. Therefore this pilot does **not** support a tail-risk advantage for individualized acquisition.

For the random policy, the table's quantiles and CVaR use the full mixture of 47 cells × six equiprobable actions; the `>5 mV` figure is the expected cell count. The comparison figure instead plots the mean action error within each cell before taking the across-cell quantile, so its random-policy tail bar (49.34 mV) answers a different, less conservative question. The JSON stores both summaries.

The post hoc per-cell oracle that may inspect each cell's true long-hold error and then choose the best of the six candidate probes reaches mean **8.37 mV**, CVaR90 **45.77 mV**, and 18/47 cells >5 mV. This is an optimistic diagnostic, **not an achievable policy**. Its CVaR90 is only 0.17 mV below the fixed ID 4 value of 45.94 mV. Including fixed ID 4 in that oracle changes CVaR90 to 45.67 mV. Thus there is little tail improvement available through choice among these probes under this analogue-update scheme, even with hindsight.

## Interpretation and decision

The probe-library gate passed, but the **acquisition-value gate failed**. The six selected probes produce distinct population response geometry, yet that diversity did not translate into a useful separation of the long-hold ambiguities represented by the current 12-analogue ensembles. A general response-diversity criterion is therefore insufficient for the actual target. The target-focused and risk-focused scores also collapsed to the same action ranking, so merely reweighting future-disagreeing pairs did not create a new effective policy here.

Do **not** send this policy to a new confirmation cohort as a proposed winning algorithm, and do not tune its pair weights or thresholds on these 47 outcomes. The next defensible algorithm-development step would require a new development population and a stronger, explicitly audited plausible-twin construction that places mass on alternative long-hold regimes while fitting B+C1+C2. Probe screening would then need to preserve **future-relevant** diversity without using the held-out cell's future response. The old ID 4 should remain in the candidate action set because it was a strong fixed reference despite failing the current response-diversity selection.

This result does not establish that risk-targeted experimental design is impossible. It establishes that the **current finite probe library + nearest-analogue ensemble + one-probe reweighting** does not beat a simple fixed probe. In particular, it does not test a calibrated posterior, a native-ODE ensemble, or robust acquisition across SAC model forms. No real-cell evidence is involved.

## Reproduction

- Library simulator: [generate_predictive_probe_library.m](generate_predictive_probe_library.m)
- Library screen: [screen_predictive_probe_library.py](screen_predictive_probe_library.py)
- Leave-one-cell-out pilot: [pilot_predictive_disagreement_design.py](pilot_predictive_disagreement_design.py)
- Numeric outputs: `human_bidirectional_output/predictive_probe_library_screen.json`, `human_bidirectional_output/predictive_disagreement_pilot_scores.json`, and `human_bidirectional_output/predictive_disagreement_pilot_cells.csv`
- Figures: `human_bidirectional_output/predictive_probe_library_screen.png` and `human_bidirectional_output/predictive_disagreement_pilot.png`
