# Cross-model static-preload test: frozen analysis rules

This protocol was written while the independent MechChem cohort was still being generated, before its cohort-level geometry was scored. It is a **different mechanical task** from the earlier short-pulse experiment. It tests whether its functional-neighborhood principle survives a published, independently structured human model under a common static-length challenge; it cannot count as replication of the exact 300 ms pulse result.

## Models and interventions

- A: existing human ToR-ORd/SAC–Land closed-loop model, 64 pre-existing cell parameters and prepaced states.
- B: published ORd–MechChem bidirectional model, 20 new Latin-hypercube cells varying eight published electrical-independent mechanics parameters. Original ORd–MechChem ODEs and author solver are used. Sarcomere length is set through its published static initial-length boundary.
- In both, `lambda = SL / SL_reference` is 1.000 (baseline), 1.050 (the only observed calibration challenge), and 1.075 and 1.100 (future challenges hidden when finding neighbors). B uses 2100 nm reference length. Pacing cycle is 1000 ms.

The model families cannot be matched cell by cell. Their shared layer is only voltage/AP, free intracellular Ca, tension, and imposed length. ORd–MechChem lacks the direct SAC of A; its mechanical feedback is through Ca–troponin binding. A's initial state is individually prepaced; B uses the public ORd initial state and fresh mechanics state. This initialization difference limits biological interpretation.

## QC and features

Input cells are excluded only for nonfinite trajectory, physiologically invalid negative Ca or tension, voltage magnitude >200 mV, or missing output window. APD is duration above −67 mV. If repolarization is absent by 999 ms, APD is right-censored at 1000 ms and the censoring count is reported. Endpoint waveform features at each load are APD, Ca peak, tension peak, voltage minimum/maximum, Ca at 200/400/600 ms, tension at 200/400/600 ms, Ca AUC, tension AUC, and tension at 999 ms. The response to a load is its feature vector minus that cell's baseline vector.

Main future outcome is the **electrical and Ca response**, excluding tension: APD, Ca peak, voltage minimum/maximum, Ca at 200/400/600 ms, and Ca AUC, for each hidden load. AP/Ca/T is secondary. This prevents a smooth length–tension curve alone from manufacturing success.

Because sustained length in A may cause an APD censored at the end of the beat, a predeclared sensitivity check uses only Ca peak, Ca at 200/400/600 ms, and Ca AUC for the future response. It answers whether the result survives removal of the censored APD dimension.

## Frozen metrics and decisions

1. Baseline observation versus baseline plus one challenge response: five-nearest-neighbor mean distance in the future-response space. Observation blocks get equal weight after robust source-population SD scaling. Future loads get equal weight. Report paired cell bootstrap 95% interval and cell-level shuffled-challenge null. Main direction required: lower future distance with challenge in B.
2. Fix 3D baseline PLS and 5D baseline-plus-challenge PLS. A-trained maps are tested on B without B future-output fitting; compare both with the zero-change predictor. Also report B-only leave-one-cell-out PLS as a **diagnostic**, not an external validation.
3. GO for a **shared functional-neighborhood principle** only if the neighbor improvement has a positive cell-bootstrap interval on the main AP/Ca target and survives the shuffled-challenge control in B. Direct transfer of A's five PLS coordinates is a stronger, separate test; failure there means the coordinates are model-specific, while the neighborhood principle may still transfer. If the main AP/Ca neighbor effect disappears, NO-GO for generality; retain the A-only observability benchmark.

No protocol or feature-selection tuning is permitted after viewing B cohort-level outcomes.
