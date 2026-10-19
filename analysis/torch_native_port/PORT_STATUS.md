# Native port status (2026-10-07)

The fixed-parameter, endocardial 43-state ToR-ORd/SAC electrical RHS and 7-state Land contraction RHS are implemented in PyTorch. `native_bidirectional.py` couples them through Ca/TnC and supports constant imposed length, the original SAC law, SAC on/off, TnC feedback on/off, length rate passed to Land, and a differentiable multiplier on the electrical stretch-sensitive increment.

Numerical validation now covers 10,100 state derivatives sampled along MATLAB trajectories and two 1000 ms whole-beat simulations at lengths 1.00 and 1.10. See `NATIVE_VALIDATION.md` for numerical results and reproducible commands.

The 50-state RHS is differentiable, and differentiation through a short explicit RK4 rollout has been checked by finite differences. The SciPy BDF solver used for whole-beat runs is a **forward-only** solver; it does not provide gradients through the 1000 ms trajectory. The current interface does not yet port the 13-parameter cell population, alternative SAC forms, or a differentiable stiff solver. Thus this is a verified native forward model plus short-horizon differentiability, not a trained UDE or a fully differentiable whole-beat platform.
