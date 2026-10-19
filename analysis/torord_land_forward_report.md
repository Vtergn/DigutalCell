# ToR-ORd → Land forward generator: first executable result

## What was run

- Electrical model: the authors' ToR-ORd MATLAB implementation (`Data/torord_official`, commit `11e463da0f083c276c0057211e476f35cc2fc2c0`), endocardial default, 1000 ms pacing cycle, three beats. The final beat supplies membrane voltage and cytosolic calcium.
- Mechanical model: the existing Physiome `Land2016.m` in intact mode, conditioned for four calcium cycles before exporting its final cycle.
- Coupling: `Cai` from ToR-ORd is converted from **mM to µM** before passing to Land. Stretch is constant during each simulation, so `dλ/dt = 0`. Electrical and calcium states are reused unchanged for every stretch of the same virtual cell. This is a **one-way** coupling; there is no mechanical feedback to AP or calcium.
- Lengths: λ = 1.000, 1.025, 1.050, 1.075, 1.100. A second run changes the same cell's `ICaL_Multiplier` to 0.8 and `Jup_Multiplier` to 1.2, demonstrating that electrical parameter inputs can also vary.

## First numerical check

| λ | Active peak tension (kPa) | Total peak tension (kPa) | Total end-cycle tension (kPa) |
|---:|---:|---:|---:|
| 1.000 | 2.676 | 2.676 | 0.002 |
| 1.025 | 5.134 | 5.670 | 0.539 |
| 1.050 | 9.963 | 11.173 | 1.214 |
| 1.075 | 19.197 | 21.252 | 2.062 |
| 1.100 | 35.372 | 38.489 | 3.129 |

Across these five loads, the same electrical trace has Vrest −88.87 mV and Vpeak 31.60 mV. Calcium ranges 0.0745–0.3858 µM and peaks at 58 ms. Tension peaks at 114–120 ms, after calcium. Active peak tension rises monotonically with stretch. The perturbed parameter example reduces peak calcium to 0.3298 µM and λ=1.1 active peak tension to 13.42 kPa. That example used only three beats after changing ionic parameters, so it is an **interface smoke test**, not a converged perturbed-cell phenotype. These are simulator outputs and qualitative checks, not independent experimental validation.

## Files and reproduction

- `analysis/torord_land_forward.m`: callable MATLAB generator. Arguments are output directory, electrical prepacing beats, ordered stretch vector, ToR-ORd `modelRunner` parameter overrides, and Land `[beta0,beta1]`.
- `analysis/torord_land_forward_output/traces.csv`: synchronized 1 ms voltage, calcium, active/passive/total tension traces at five stretches.
- `analysis/torord_land_forward_output/summary.csv`: load-specific peak and end-cycle features.
- `analysis/torord_land_forward_output/metadata.json`: parameter and unit provenance.
- `analysis/torord_land_forward_output/ap_ca_active_tension.png`: waveform check.
- `analysis/torord_land_forward_output/perturbed/`: second parameter identity, evaluated at λ=1, 1.05, 1.1.

Run from MATLAB with `addpath('D:/MyProjects/AI细胞/analysis'); torord_land_forward` or pass custom arguments, e.g. `torord_land_forward('D:/MyProjects/AI细胞/analysis/custom',3,[1 1.05 1.10],struct('ICaL_Multiplier',0.8))`.

## Scientific boundary before an inverse benchmark

This is not yet a reproduction of the Strocchi 27-parameter ToR-ORd–Land ensemble. The author ToR-ORd code and downloaded Land implementation do not expose those 27 variables through the same interface; the current generator exposes the standard ToR-ORd runner parameters and Land's β0/β1. The Strocchi tables also use **active tension** features, so their `Trest` should not be compared with this run's *total* end-cycle tension, which includes passive tension. The exact Strocchi parameterization and feature extraction must be reconstructed before claiming a matched benchmark.

The five stretch conditions now establish that new, same-parameter paired AP/Ca/T traces can be generated. They do not establish that an inverse twin predicts an unseen load better than direct regression; a direct model supplied with stretch as an input is a fair comparison. In particular, predicting λ=1.05 after observing λ=1.00 and 1.10 is **interpolation in load**, even though that exact condition is held out. A separate outside-range test is needed for extrapolation. Also, the one-way model cannot answer questions about mechanical feedback into AP or calcium. Isotonic load requires a length/force-control equation and has not been implemented here.
