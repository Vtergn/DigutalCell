# Conditioned neural virtual-cell decoder

This is the second prototype. It uses 47 valid cells from the existing
ToR-ORd/SAC–Land development cohort, with a cell-held-out split (35 train,
12 test) and six mechanical protocols. The decoder predicts the protocol
response relative to that cell's baseline trajectory, conditioned on time,
mechanical length, pulse protocol, and the baseline V/Ca/T state.

Held-out-cell errors at a 10 ms time step:

- Voltage MAE: 0.994 mV
- Ca MAE: 0.03299 µM
- Tension MAE: 0.158 kPa

The baseline-conditioned delta formulation is substantially more stable than
the first recursive state-transition prototype, whose long rollout accumulated
error. This is still a neural surrogate, not a replacement for the mechanistic
ODE and not an independent biological validation. The network sees the same
six protocol families during training; it has not yet been tested on a wholly
unseen protocol family or another model.

Files:

- `conditioned_virtual_cell.gif`: rendered held-out virtual cell
- `metrics.json`: held-out metrics
- `decoder.pt`: network weights
- `train_conditioned_behavior_decoder.py`: reproduction script
