"""Source-informed reduced EDS model for an identifiability gate.

The single-shell Clausius–Mossotti expression follows Du et al., Eq. (4),
specialized to a spherical cell (A1=A2=1/3). The measured response is a
small-deformation proxy proportional to Re(f_CM)/mu. It is NOT the paper's
full geometry-dependent forward/inverse model.
"""

from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
RADIUS_M = 3.9e-6
MEMBRANE_M = 4.5e-9
EPS_MEDIUM = 78.0  # assumption for aqueous measurement solution
SIGMA_MEDIUM = 0.01  # S/m, Du et al. measuring solution
EPS_CYTO = 59.0  # Du et al. fixed value
SIGMA_MEM = 1e-6  # S/m, Du et al. fixed value
RHO = ((RADIUS_M - MEMBRANE_M) / RADIUS_M) ** 3
A = 1.0 / 3.0


def cm_factor(freq_hz: np.ndarray, eps_mem: float, sigma_cyto: float,
              sigma_mem: float = SIGMA_MEM) -> np.ndarray:
    w = 2 * np.pi * np.asarray(freq_hz, dtype=float)
    m = EPS_MEDIUM * EPS0 - 1j * SIGMA_MEDIUM / w
    s = eps_mem * EPS0 - 1j * sigma_mem / w
    c = EPS_CYTO * EPS0 - 1j * sigma_cyto / w
    numerator = (s - m) * (s + A * (c - s)) + RHO * (c - s) * (s - A * (s - m))
    denominator = (m + A * (s - m)) * (s + A * (c - s)) + RHO * A * (1 - A) * (c - s) * (s - m)
    return numerator / (3 * denominator)


def response(freq_hz: np.ndarray, log_parameters: np.ndarray) -> np.ndarray:
    """Dimensionless deformation proxy; mu in microN/m, log parameters."""
    eps_mem, sigma_cyto, mu = np.exp(log_parameters)
    return np.real(cm_factor(freq_hz, eps_mem, sigma_cyto)) / mu


def jacobian(freq_hz: np.ndarray, log_parameters: np.ndarray, step: float = 1e-4) -> np.ndarray:
    columns = []
    for j in range(3):
        shift = np.zeros(3)
        shift[j] = step
        columns.append((response(freq_hz, log_parameters + shift) - response(freq_hz, log_parameters - shift)) / (2 * step))
    return np.column_stack(columns)
