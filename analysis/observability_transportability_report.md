# What a force–current digital twin can predict

2026-09-28. This extends the [threshold and observability audit](threshold_exponent_and_observability_report.md). Code: `digital_twin_transportability_audit.py`, `dynamic_label_swap_audit.py`, and the expanded `mechanotransduction_observability_framework.py`. Numerical currents here are **synthetic**. Duke contributes only the observed peak-force range of 34 responding Piezo1/Piezo2 cells; no Duke cell is assigned a fitted local tension, channel count, or time constant.

## 1. Observational equivalence is not intervention equivalence

The static sigmoid model \(I=G\sigma[(T_0+\lambda F-T_{50})/k]\) has an exact two-parameter tension-coordinate symmetry. I used a reference parameter set \(T_0=1.3\) mN/m, \(\lambda=0.007\) mN/m/nN, \(T_{50}=2.7\) mN/m, \(k=0.7\) mN/m and \(G=200\) pA, then generated 2,501 equivalent parameter sets with tension scale \(a=0.5\)–2 and offset \(b=-0.5\)–0.5 mN/m. Across 0–650 nN, the maximum discrepancy in their ordinary force–current curves is \(1.1\times10^{-13}\) pA: numerically zero.

But the twins answer different **physical interventions** differently:

| Challenge | Predictions among equivalent models | Why |
|---|---:|---|
| Increase AFM force from 200 to 250 nN | 124.49 pA in every model | Still only probes the same force-domain function |
| Add a calibrated local \(+0.3\) mN/m tension at 200 nN | 110.67–140.41 pA | A physical tension increment anchors its unit/scale |
| Clamp local absolute tension at 3.0 mN/m | 22.38–199.57 pA | Absolute tension also anchors the zero point |

The last two ranges are **illustrations conditional on chosen plausible-scale parameters and interventions**, not uncertainty intervals for real cells. On the 34 observed Duke peak-force values used as challenge inputs, the median prediction spread under the \(+0.3\) mN/m example is 9.1 pA (cell-level IQR 5.1–27.7 pA); saturation at larger forces reduces the spread. The biological lesson is narrower and exact: a model that predicts all future *force-only* stimuli equally well may still be unable to predict an intervention specified in **absolute membrane-tension units**.

This also clarifies what a claimed “digital twin” predicts. The quotient parameters \(F_{50},s_F,G\) may support a **force-domain emulator within the same cellular context**. A twin that predicts tension-targeted perturbations or transports across altered force-transmission geometry needs calibrated mechanical information or justified transported priors. Merely collecting more of the same \(F,I\) sweeps cannot break the exact symmetry. Exactly known, transferable \(T_{50}\) **and** \(k\) could also anchor it in principle; published patch values from different geometries are uncertain constraints, not exact same-cell identities.

## 2. Dynamic timing can have a separate label ambiguity

In a minimal linearized cascade,

\[
\tau_m\dot m=F-m,\qquad \tau_g\dot q=m-q,\qquad I=Gq,
\]

the force-to-current transfer function is \(G/[(1+\tau_m s)(1+\tau_g s)]\). Exchanging \(\tau_m\) and \(\tau_g\) **leaves the entire current waveform unchanged for any force input**, assuming the same zero initial state. Thus dense synchronized \(F,I\) recordings can estimate two poles yet fail to label which pole is mechanical relaxation and which is gating. This is a property of this linearized series model, not a theorem for every nonlinear PIEZO mechanism.

A 1-ms-grid synthetic pulse with \(\tau_m,\tau_g=(2,10)\) ms and its swapped model gave a maximum current difference of \(2.8\times10^{-14}\) pA. Their intermediate mechanical states differed by as much as 0.535 on a unit scale. If that state could be measured with SD 0.05, the expected squared model separation would be about 2,132 at 1-ms sampling, but only 0.60 when the example pulse is sampled at 40-ms intervals. Across 40 different pulse phases, the 40-ms separation ranged 0.27–229; **sampling phase matters greatly**. This illustration explains why a slower mechanical proxy need not resolve a few-ms mechanical/gating split even though it measures the right kind of variable. It does not assert that ETH's measured Flipper-TR signal is this model state or has SD 0.05.

An unrestricted learned latent ODE has an even broader coordinate freedom: for any invertible \(z=\phi(T)\), changing its dynamics to \(\dot z=\phi'(T)f(T,F)\) and its readout to \(h(\phi^{-1}(z))\) preserves every \(F\to I\) prediction. The checked example \(z=T^2\) applies for positive tension. Therefore a neural latent state cannot acquire a physical tension unit merely by fitting waveform data. A measured intermediate or a restrictive physical model is required for that interpretation.

## 3. What this contributes, and what it does not

The general ideas of structural identifiability and symmetry are established in systems biology; this work does **not** claim to invent them. See, for example, [Villaverde et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6685009/) and [Finding and Breaking Lie Symmetries, 2020](https://www.mdpi.com/2073-8994/12/3/469). The potentially useful project-specific contribution is to make them operational for a force–tension–gating–current digital twin: distinguish the predictions that remain invariant, exhibit the interventions for which equivalent models diverge, and specify the intermediate measurement and sampling needed to label mechanical versus electrical dynamics.

The current material is a **formal model audit plus synthetic benchmark**, supported by real Duke force ranges and the separately audited ETH/eLife scales. It is not a demonstrated biological mechanism and cannot establish a real cell's \(T_0\), \(\lambda\), \(\tau_m\), or \(\tau_g\). A publication claim would need a systematic class of mechanotransduction models, robustness to model mismatch and measurement error, and ideally an independent same-cell mechanical/electrical dataset. The two exact counterexamples already justify a firm modeling rule: report force-domain predictive parameters as such, and do not rename an unobserved latent state “membrane tension” or an unlabeled waveform pole “channel inactivation.”
