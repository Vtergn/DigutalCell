# PyTorch native human electromechanical model

`native_bidirectional.py` contains a 50-state ToR-ORd/SAC–Land coupled RHS: 43 electrical states and 7 contraction states. The present port represents the fixed endocardial parameterization and the original SAC feedback law. `electrical_gating_partial.py` is retained under its historical filename but now implements the full 43-state electrical RHS. `land2016_torch.py` implements the 7-state contraction module.

The 50-state RHS and short explicit RK4 rollouts are differentiable. TorchScript tracing accelerates the RHS on CPU. Whole-beat simulation currently uses a forward-only SciPy BDF solve because the equations are stiff.

See [validation and timings](NATIVE_VALIDATION.md) and [scope and remaining work](PORT_STATUS.md). The comparison figure is `full_beat_validation.png`.
