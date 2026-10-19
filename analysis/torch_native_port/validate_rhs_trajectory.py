"""Compare the full PyTorch RHS with MATLAB along native trajectories."""
from pathlib import Path
import json

import numpy as np
import torch
from scipy.io import loadmat

from analysis.torch_native_port.native_bidirectional import native_rhs

HERE = Path(__file__).resolve().parent


def main():
    z = loadmat(HERE / "trajectory_reference.mat")
    times = z["times"].ravel()
    lengths = z["lengths"].ravel()
    states = z["states"]
    reference = z["derivs"]
    predicted = np.empty_like(reference)
    for j, length in enumerate(lengths):
        for k, t in enumerate(times):
            y = torch.tensor(states[k,:,j],dtype=torch.float64)
            predicted[k,:,j] = native_rhs(y,float(t),float(length))[0].detach().numpy()
    error = np.abs(predicted-reference)
    report = {
        "samples": int(error.size),
        "states": int(states.shape[1]),
        "time_points_per_condition": int(states.shape[0]),
        "lengths": lengths.tolist(),
        "mean_absolute_derivative_error": float(error.mean()),
        "p99_absolute_derivative_error": float(np.quantile(error,.99)),
        "max_absolute_derivative_error": float(error.max()),
        "worst_index_time_state_condition": [int(v) for v in np.unravel_index(error.argmax(),error.shape)],
    }
    (HERE/"rhs_trajectory_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
