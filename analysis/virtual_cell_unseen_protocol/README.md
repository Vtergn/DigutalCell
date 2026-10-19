# Blind protocol-family test

The long-hold protocol (protocol 3) was completely excluded from training.
The decoder used five other protocols from 35 cells and was evaluated on 12
held-out cells. It received the held-out cell's baseline V/Ca/T trajectory and
the prescribed length trajectory, then reconstructed the long-hold response.

Results:

- Voltage MAE: 3.09 mV
- Ca MAE: 0.0246 µM
- Tension MAE: 0.111 kPa

This is a meaningful first generalization test: the model did not see any
long-hold target waveform during training. It is still not a strong
mechanistic claim. The test uses the same simulator family, the baseline
trajectory is supplied as an input, and only one protocol family is held out.
The native ODE remains the scientific reference model.
