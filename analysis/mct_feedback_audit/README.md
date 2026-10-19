# Mechanical-feedback resource audit and native-model ablation

This folder extends the virtual-cardiomyocyte benchmark with a **separate rabbit closed-loop reference domain**. The human ToR-ORd→Land cohort remains a one-way AP→Ca→tension system. No rabbit MCT coefficient is transferred to its human ion model.

## Experimental archive

`audit_mct_experiment.py` reads `Data/doi_10_25338_B8T935__v20220607/Experimental_Data.zip` without extracting it. The deposited `README.pdf` identifies rabbit left-ventricular myocytes. The archive provides one load-free and one in-gel example trace each for Fura-2 ratio and sarcomere length, at 4-ms sampling over 6.288 s, with raw and smoothed copies. There are no cell IDs or replicate tables in the archive. The gel example has the higher Fura-2 peak (3.07 versus 2.03) and the smaller fractional sarcomere excursion (~10.4% versus ~12.7%); that paired *direction* matches the native MCT-on model's high-load response, but the example traces cannot establish an effect size or independent parameter validation. `archive_audit.json` and `rabbit_load_feedback_traces.png` are descriptive checks only; the Fura-2 ratio is not absolute intracellular calcium.

## Native MCT2024 reference

`prepare_mct2024_reference.py` creates `mct2024_single_condition.m` from the downloaded author code. It:

1. Replaces the bundled many-PCL, 300-beat sweep with one PCL=1000 ms and a caller-specified beat count.
2. Uses `yfinal_2000.mat`, the initial-state file actually present in the downloaded release. The original script asks for an absent `yfinal_3000.mat`.
3. Returns voltage, calcium, mechanics, gel stress, and L-type Ca permeability multiplier as numerical traces instead of writing dozens of text files.
4. Exposes an MCT gain. Gain=1 is algebraically the original gel-stress→Ca-permeability equation; gain=0 fixes that multiplier to 1 and leaves all other native ODEs in place.

The copied source code is a local research reference. Read the author's repository and model paper before reusing or redistributing it. We checked gain=1 by comparing its first eight beats with a separately run original-equation version: the maximum Ca difference was 1.03×10⁻⁹ µM.

The four 20-beat simulations use gel crosslinker concentration `CL=0.02` or `0.10`, crossed with MCT gain 0 or 1, all from the same supplied initial state. The quantitative results are `native_mct_ablation.json`; the last-beat waveforms are in `native_mct_ablation.png`. Calcium peak drift between beats 19 and 20 is below 0.12% in all four conditions.

At CL=0.02→0.10, the native model gives Ca peak 0.426→0.673 µM and APD above −67 mV 264→351 ms with MCT on. With MCT off these stay at approximately 0.361 µM and 242 ms. The high-load condition makes the on-versus-off Ca/APD gap about 4.9× larger than under low load. This is a model ablation at one parameter state, not an estimated experimental effect or a population-level virtual-cell result.

## Reproduction

From the project root, run:

```powershell
.venv_tcn\Scripts\python.exe analysis\audit_mct_experiment.py
.venv_tcn\Scripts\python.exe analysis\prepare_mct2024_reference.py
```

Then invoke `mct2024_single_condition(CL,20,init_file,gain)` from MATLAB with the audit folder and downloaded MCT folder on the MATLAB path; save each resulting structure as a v7.3 `.mat`. Finally run:

```powershell
.venv_tcn\Scripts\python.exe analysis\score_mct2024_reference.py
```

## Interpretation boundary

The archive cannot identify patient/cell-specific mechanical-feedback parameters. The two model domains use different species, electrophysiology, contraction, and load definitions. Their outputs can support *within-domain* challenge/ablation comparisons, but should not be pooled into one training table or treated as one calibrated closed-loop human simulator. A future integrated human model needs a measured or independently justified mapping from its force/strain state to the calcium-current feedback term.
