# Minimal module falsification before a hybrid Neural ODE

## Scope and frozen inputs

This audit tests three narrowly defined claims. It does not refit Duke's force-to-current model or use Duke perturbations to tune the external modules. Scripts are `module_falsification_p1.py` and `module_falsification_p2.py`. The PIEZO1 source is the published [wedge/inactivation study](https://www.nature.com/articles/s41467-026-76927-0); the PIEZO2 mechanism comparator is the published [FLNB–actin study](https://www.nature.com/articles/s41586-026-10182-7). Local Source Data were extracted in the earlier wedge and FLNB audits.

### P1-1. Does a mechanical relaxation descriptor generalize across loading speeds?

The published AFM speed table contains force-relaxation time constants and retained-force fractions for seven cells at six prescribed speeds, with one missing cell–speed point (41 records). It does **not** contain the raw force-versus-time trajectories needed to fit or test a standard linear solid or state ODE. I therefore tested the strongest claim these values allow: leave one entire speed out; use the other speeds from each measured cell to estimate a cell intercept and a slope shared across cells; predict the unseen speed. The comparator predicts each cell's speed-independent mean. The response variables are log relaxation time and logit retained-force fraction. No current is involved.

| Endpoint, held-out speed | Cell-constant median absolute error | Shared speed slope median absolute error |
| --- | ---: | ---: |
| Relaxation time, ms | 45.89 | **7.80** |
| Force remaining, percentage points | 9.82 | **3.36** |

The estimated log-speed slope of relaxation time remains −0.515 to −0.560 across the six folds. This supports a stable *within-cell, cross-speed descriptor relation* for this protocol. It does not establish a generalizable latent mechanical state, a full relaxation waveform, a cross-cell prediction rule, or a transferable AFM-to-membrane-tension operator. A simple speed law already explains the available summary data, so these data alone do not motivate a neural mechanical ODE.

### P1-2. Does one discrete history state beat simpler pulse controls?

For each measured interval (50, 100, 200, 500, 1000 ms), I left out one source-data column, fit the other columns **at that interval only**, then predicted the held-out normalized responses to pulses 2–6 starting at pulse 1 = 1. The tested recurrence is `r[n+1] = a_interval r[n] + b_interval`, with `a,b` bounded to [0,1]. It was compared with a no-history ratio of 1, a fitted constant ratio after pulse 1, and a separate five-pulse template. There was no interval interpolation or recovery-time fit.

| Interval, ms | Source columns | No history median MAE | Post-pulse constant | One state |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 9 | 0.269 | 0.180 | **0.166** |
| 100 | 7 | 0.155 | 0.156 | **0.131** |
| 200 | 8 | 0.168 | 0.134 | **0.115** |
| 500 | 6 | 0.217 | **0.098** | 0.101 |
| 1000 | 8 | **0.151** | 0.152 | 0.188 |

The state model beats the post-pulse constant for 24 of 38 held-out columns; the 50-ms mean error is essentially tied, and the model is worse at 500 and 1000 ms by median error. One near-zero 1000-ms trace strongly affects that group; removing it *as a sensitivity check* gives state 0.117 versus constant 0.128, still a small comparison based on only seven columns. The same patch may have been measured at multiple intervals, but cross-interval patch IDs are absent. Interval also follows a fixed experimental order. These numbers are descriptive, not 38 independent biological replicates and not evidence for a continuous recovery equation. History dependence is present in the source experiment, but a transferable one-dimensional state has not passed a stable predictive test.

### P2. Is there a weak PIEZO2-selective cytoskeletal perturbation signal in Duke?

For each date containing both conditions and responsive cells, I compared within-date medians from the frozen Duke CytoD summary and the already frozen signed-current waveform metrics. PIEZO2 uses DMSO versus CytoD; PIEZO1 uses its available untreated versus CytoD control. I did not pool dates or fit a cross-channel interaction because the channel experiments took place on different dates, and the PIEZO1 comparator is not explicitly DMSO-matched.

| Frozen endpoint | PIEZO1: 6 dates | PIEZO2: 3 dates |
| --- | ---: | ---: |
| Geometric mean CytoD/control work-threshold ratio, responders | 0.938 | 0.565 |
| Dates with a lower work threshold | 4/6 | 2/3 |
| Dates with a higher selected peak current | 4/6 | 3/3 |
| Dates with a lower peak force and work | 3/6 and 4/6 | 3/3 and 3/3 |
| Dates with lower 20-ms post-peak current retention | 3/6 | 2/3 |
| Responders, control → treated | 12/12 → 12/12 | 14/19 → 18/18 |

There is a tentative PIEZO2-selective threshold/responder signal, consistent with earlier date-stratified and censoring audits. The amplitude comparison is especially difficult to interpret: PIEZO2 treated cells had larger selected peak current in all three dates despite lower selected peak force and work, and the control group includes nonresponders. The waveform-shape sign is mixed. CytoD disrupts actin broadly, whereas FLNB knockout changes a specific tether; they need not produce the same threshold direction. This audit does **not** establish a Duke FLNB-tether parameter or a channel-by-treatment causal interaction.

## Go/no-go decision

**No-go for a hybrid Neural ODE with current public summary data.** P1-1 passes only a cross-speed *descriptor* check, not a dynamic-state check. P1-2 finds no stable gain for one transferable history state over simple controls. P2 provides a useful perturbation direction for a channel-specific mechanical architecture, but no time-resolved tether equation. The supported model remains modular: channel-specific mechanical encoders, separately constrained gating, a conservative history annotation, and explicit cell calibration/identifiability limits. A future NODE would require time-resolved force/current under randomized loading speeds and randomized inter-pulse intervals, with patch identity retained. The first test should be held-out protocols, not another Duke parameter sweep.

## Reproducibility

Run with `.venv_tcn/Scripts/python.exe analysis/module_falsification_p1.py` and `.venv_tcn/Scripts/python.exe analysis/module_falsification_p2.py` from the project root. Detailed outputs: `module_p1_falsification_results.json`, `module_p1_speed_holdout_scores.csv`, `module_p1_history_holdout_scores.csv`, `module_p1_speed_holdout.png`, `module_p2_cytod_results.json`, and `module_p2_cytod_date_audit.csv` in this directory.
