# Native PyTorch port: numerical validation and optimization

All comparisons below use CPU, `torch.float64`, the fixed endocardial model, the same saved pre-paced initial state, and the original SAC feedback form. The MATLAB reference uses the existing native model. Timings are measurements on this machine, not portable performance guarantees.

| Check | Result |
| --- | ---: |
| MATLAB vs PyTorch RHS, 50 states × 101 time points × 2 lengths | mean absolute derivative difference `1.80e-8`; 99th percentile `2.13e-7`; maximum `1.11e-5` |
| Eager RHS | `8.29 ms/call` |
| TorchScript traced RHS | `4.06 ms/call` (`2.04×` speedup) |
| Traced vs eager RHS, maximum output difference | `9.97e-9` |
| 2 ms explicit RK4 trajectory vs MATLAB, maximum voltage difference | `0.00809 mV` |
| 2 ms forward SciPy BDF trajectory vs MATLAB, maximum voltage difference | `4.10e-5 mV` |
| 1000 ms beat, length 1.00, voltage/Ca/active-tension MAE | `0.01144 mV` / `9.83e-5 µM` / `0.01123 kPa` |
| 1000 ms beat, length 1.10, voltage/Ca/active-tension MAE | `0.01495 mV` / `6.48e-5 µM` / `0.02520 kPa` |

The 1000 ms forward BDF runs took `25.7 s` and `21.0 s`, respectively. These errors compare independent adaptive numerical solvers, so trajectory errors are larger than pointwise RHS errors. They establish agreement with this MATLAB implementation, not experimental validation of its biological assumptions.

Autograd through a 0.1 ms RK4 rollout gave `dV/d(sac_gain) = 0.112048031203 mV`; central finite differences gave `0.112048031369 mV` (relative difference `1.48e-9`). MATLAB/PyTorch gain-perturbation RHS derivatives also matched to `1.24e-17` maximum absolute error. TorchScript preserved dependence on the time and length inputs. The traced state-gradient maximum absolute difference from eager was `1.30e-5`; this was not used to assert whole-beat gradient fidelity.

## Reproduce

Run from the project root after the MATLAB reference files have been generated:

```powershell
python -m analysis.torch_native_port.validate_rhs_trajectory
python -m analysis.torch_native_port.benchmark_compiled_rhs
python -m analysis.torch_native_port.validate_compiled_trajectory
python -m analysis.torch_native_port.benchmark_stiff_solver
python -m analysis.torch_native_port.validate_differentiable_rollout
python -m analysis.torch_native_port.validate_full_beat 1.0
python -m analysis.torch_native_port.validate_full_beat 1.1
python -m analysis.torch_native_port.plot_full_beat_validation
```

The whole-beat plot is `full_beat_validation.png` (also PDF). Individual machine-readable results are in the corresponding JSON and NPZ files. BDF solves in this workflow use a NumPy callback and **are not differentiable**; only the PyTorch RHS and explicit short RK4 rollout are differentiated.
