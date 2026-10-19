# Native reduced UDE gate

## Purpose

This is the first test that trains a derivative residual through an explicit RK4 rollout. It is deliberately **not** a full ToR-ORd/SAC–Land UDE: the prior field is a linear three-output reduced vector field fitted from the existing trajectories.

## Configuration

- 47 valid virtual cells; 20 development cells used for this quick gate and 12 held-out cells.
- Three seen protocols (0, 1, 2); protocol 3 held out.
- 3 normalized state outputs: V, Ca, tension.
- 20 training epochs; 2,739 neural parameters; CPU.
- Each epoch backpropagates through roughly 100 RK4 steps for each cell and protocol.

## Results

`json
{
  "train_cells": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ],
  "test_cells": [
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46
  ],
  "held_protocol": 3,
  "epochs": 20,
  "elapsed_seconds": 18.853231500135735,
  "n_parameters": 2739,
  "prior_MAE_V_Ca_T": [
    1365.802001953125,
    6.830248832702637,
    1425.7784423828125
  ],
  "native_reduced_ude_MAE_V_Ca_T": [
    391.4999084472656,
    388.6904602050781,
    438.2384338378906
  ],
  "prior_overall_MAE": 932.8035888671875,
  "native_reduced_ude_overall_MAE": 406.1430358886719,
  "note": "True RK4 rollout of a reduced vector field; not the full native ToR-ORd/SAC-Land RHS."
}
`

The run took about 18.9 s for only 20 epochs. This is already much slower than the earlier direct-output diagnostics. The reduced model is numerically poor in this quick gate: the linear prior itself is unstable over the long rollout, and the bounded residual reduces but does not solve the error. This is a **go/no-go warning**, not evidence that a full UDE works.

## Interpretation

The test confirms why the earlier experiments completed quickly: they optimized direct waveform outputs or used supplied trajectories. A native UDE must differentiate through the integrator and can become unstable over long horizons. The next technically valid step is to port a small native feedback subsystem from human_bidirectional_rhs.m into a differentiable solver, validate it against MATLAB first, then add a residual. Porting the full 43-state RHS before this reduced validation would be premature.
