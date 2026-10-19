# Neural virtual cardiomyocyte prototype

This prototype tests the “neural network reconstruction + rendered cell” idea
on the existing human bidirectional model outputs.

## What it does

- Reads the eight existing early/late pulse traces for the four feedback configurations.
- Trains a small multilayer neural decoder on time, imposed length, pulse onset,
  SAC state, and TnC-Ca state.
- Reconstructs membrane voltage, intracellular Ca, and TRPN state.
- Renders an animated cell avatar whose width follows the imposed length,
  membrane colour follows voltage, and intracellular colour follows Ca.

## Result

Training reconstruction RMSE:

- Voltage: 3.43 mV
- Ca: 0.00604 µM
- TRPN: 0.00399 (fraction units)

This is a **representation and visualization prototype**, not an independent
generalization test. The network was trained on the same eight traces used for
the demonstration. It shows that the visual interface can be built, but it
does not yet establish a predictive neural digital twin.

## Files

- `virtual_cell_demo.gif`: animated virtual-cell display
- `metrics.json`: reconstruction metrics
- `decoder.pt`: trained decoder weights
- `prototype_virtual_cell.py`: reproducible script

The next scientifically meaningful test is to split by cell and protocol,
train on a larger virtual-cell cohort, and evaluate blind future trajectories
against the native ToR-ORd/SAC–Land model.
