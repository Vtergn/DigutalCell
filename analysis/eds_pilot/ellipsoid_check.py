"""Source-equation consistency check with an explicit volume-conserving closure.

Uses Du et al. Eq. (4), (5), (7), (11). The paper measures a and b at each
frequency, then computes c from cell volume; here a prospective forward
calculation needs a closure, so b and c are both scaled by lambda**(-1/2).
This closure is an assumption and is not an exact reproduction of the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares

from model import EPS0, EPS_MEDIUM, EPS_CYTO, SIGMA_MEDIUM, SIGMA_MEM, MEMBRANE_M

A0, B0, C0 = 7.8e-6, 7.8e-6, 2.7e-6  # diameters; source Table 1 ranges
FIELD_GRADIENT = 7e14
OUT = Path(__file__).resolve().parent / "results"


def depolarization_major(diameters: np.ndarray) -> float:
    axes = np.asarray(diameters) / 2
    a, b, c = axes
    # Scale by a^2 to keep quadrature in a numerically benign dimensionless form.
    ratios = axes / a
    aa, bb, cc = ratios
    def integrand_scaled(u):
        s = u / (1-u)
        return (1/(1-u)**2) / ((s+aa*aa)*np.sqrt((s+aa*aa)*(s+bb*bb)*(s+cc*cc)))
    result, _ = quad(integrand_scaled, 0, 1, epsabs=1e-9)
    return float(aa * bb * cc * result / 2)


def geometry(lam: float) -> np.ndarray:
    return np.array([A0 * lam, B0 / np.sqrt(lam), C0 / np.sqrt(lam)])


def cm_ellipsoid(freq_hz: float, eps_mem: float, sigma_cyto: float, lam: float) -> complex:
    outer = geometry(lam)
    inner = outer - 2 * MEMBRANE_M  # diameters shrink by twice radial thickness
    a1, a2 = depolarization_major(outer), depolarization_major(inner)
    rho = np.prod(inner / outer)
    w = 2 * np.pi * freq_hz
    m = EPS_MEDIUM * EPS0 - 1j * SIGMA_MEDIUM / w
    s = eps_mem * EPS0 - 1j * SIGMA_MEM / w
    c = EPS_CYTO * EPS0 - 1j * sigma_cyto / w
    num = (s-m)*(s+a1*(c-s)) + rho*(c-s)*(s-a1*(s-m))
    den = (m+a1*(s-m))*(s+a1*(c-s)) + rho*a2*(1-a2)*(c-s)*(s-m)
    return num / (3*den)


def force_balance(lam: float, freq_hz: float, eps_mem: float,
                  sigma_cyto: float, mu_micro_n_m: float) -> float:
    a, b, c = geometry(lam)
    fcm = np.real(cm_ellipsoid(freq_hz, eps_mem, sigma_cyto, lam))
    electrical_force = np.pi/4*a*b*c * EPS_MEDIUM*EPS0 * fcm * FIELD_GRADIENT * (a/1e-6)**(-.86)
    mechanical_force = mu_micro_n_m*1e-6 * b * (lam*lam-lam**-2)
    return electrical_force - mechanical_force


def elongation(freq_hz: float, eps_mem: float, sigma_cyto: float,
               mu_micro_n_m: float) -> float:
    low, high = 1.0, 3.0
    if force_balance(low, freq_hz, eps_mem, sigma_cyto, mu_micro_n_m) <= 0:
        return low
    while force_balance(high, freq_hz, eps_mem, sigma_cyto, mu_micro_n_m) > 0:
        high *= 1.5
        if high > 10:
            raise RuntimeError("No stable forward root in physical search range")
    return brentq(force_balance, low, high,
                  args=(freq_hz, eps_mem, sigma_cyto, mu_micro_n_m), xtol=1e-8)


def main():
    frequencies = np.geomspace(1e5, 15e6, 7)
    truth = np.array([4.7, .65, 2.5])
    observed = np.array([elongation(f, *truth) for f in frequencies])
    # Inverse of the same declared geometric closure, noise-free consistency.
    fit = least_squares(lambda p: np.array([elongation(f, *np.exp(p)) for f in frequencies])-observed,
                        np.log([5.5,.8,3.0]), bounds=(np.log([.1,.1,.1]),np.log([30,3,30])),
                        diff_step=1e-3, max_nfev=30, xtol=1e-9, ftol=1e-9)
    payload = {"scope":"Du Eq4/5/7/11 with assumed volume-conserving transverse closure",
               "frequencies_mhz":(frequencies/1e6).tolist(),
               "elongation":observed.tolist(),
               "true_parameters":truth.tolist(),
               "recovered_parameters":np.exp(fit.x).tolist(),
               "max_absolute_elongation_residual":float(np.max(np.abs(fit.fun))),
               "volume_relative_variation":float(np.ptp(np.prod([geometry(l) for l in observed],axis=1)) / np.prod(geometry(1)))}
    (OUT/"ellipsoid_consistency.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
