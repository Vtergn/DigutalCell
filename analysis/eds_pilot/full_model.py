"""Vectorized ellipsoid EDS equations with explicit forward shape closures.

Du et al. Eq. 4, 5, 7 and 11 are implemented here. Original EDS inference
measures a and b at every frequency and derives c from the measured cell
volume. Forward synthesis additionally needs a rule for b(lambda); the two
rules below are declared assumptions, not equations inferred from the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares

from model import EPS0, EPS_MEDIUM, EPS_CYTO, SIGMA_MEDIUM, SIGMA_MEM, MEMBRANE_M

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)
_U = (_NODES + 1.0) / 2.0
_W = _WEIGHTS / 2.0


def depolarization_major(diameters: np.ndarray) -> np.ndarray:
    """Eq. 5 via fixed Gauss-Legendre quadrature; last axis has length 3."""
    dims = np.asarray(diameters, dtype=float)
    ratios = dims / dims[..., :1]
    rb, rc = ratios[..., 1], ratios[..., 2]
    s = _U / (1 - _U)
    integrand = (1 - _U) ** -2 / ((s + 1.0) *
        np.sqrt((s + 1.0) * (s + rb[..., None] ** 2) * (s + rc[..., None] ** 2)))
    return rb * rc * np.sum(_W * integrand, axis=-1) / 2.0


def cm_factor(freq_hz: np.ndarray, dims_m: np.ndarray, eps_mem: float,
              sigma_cyto: float, sigma_mem: float = SIGMA_MEM) -> np.ndarray:
    freq = np.asarray(freq_hz, dtype=float)
    dims = np.asarray(dims_m, dtype=float)
    inner = dims - 2 * MEMBRANE_M
    a1 = depolarization_major(dims)
    a2 = depolarization_major(inner)
    rho = np.prod(inner / dims, axis=-1)
    w = 2 * np.pi * freq
    m = EPS_MEDIUM * EPS0 - 1j * SIGMA_MEDIUM / w
    shell = eps_mem * EPS0 - 1j * sigma_mem / w
    core = EPS_CYTO * EPS0 - 1j * sigma_cyto / w
    num = (shell-m)*(shell+a1*(core-shell)) + rho*(core-shell)*(shell-a1*(shell-m))
    den = (m+a1*(shell-m))*(shell+a1*(core-shell)) + rho*a2*(1-a2)*(core-shell)*(shell-m)
    return num / (3*den)


def dimensions(lam: np.ndarray, a0: float, b0: float, c0: float,
               closure: str = "balanced") -> np.ndarray:
    lam = np.asarray(lam)
    if closure == "balanced":
        b, c = b0 / np.sqrt(lam), c0 / np.sqrt(lam)
    elif closure == "fixed_thickness":
        b, c = b0 / lam, np.broadcast_to(c0, lam.shape)
    else:
        raise ValueError(closure)
    return np.stack(np.broadcast_arrays(a0*lam, b, c), axis=-1)


def electrical_force(freq_hz: np.ndarray, dims_m: np.ndarray,
                     eps_mem: float, sigma_cyto: float,
                     field_gain: float = 1.0,
                     sigma_mem: float = SIGMA_MEM) -> np.ndarray:
    dims = np.asarray(dims_m)
    fcm = np.real(cm_factor(freq_hz, dims, eps_mem, sigma_cyto, sigma_mem))
    return (np.pi/4 * np.prod(dims, axis=-1) * EPS_MEDIUM*EPS0 * fcm *
            7e14 * (dims[..., 0]/1e-6)**(-.86) * field_gain)


def mechanical_force(lam: np.ndarray, b_m: np.ndarray, mu_micro_n_m: float,
                     nonlinear_kappa: float = 0.0,
                     mechanics_form: str = "linear_plus") -> np.ndarray:
    lam = np.asarray(lam)
    strain = lam - 1.0
    if mechanics_form == "linear_plus":
        factor = 1.0 + nonlinear_kappa*strain
    elif mechanics_form == "exponential":
        factor = np.exp(nonlinear_kappa*strain)
    elif mechanics_form == "saturating":
        factor = 1.0 + nonlinear_kappa*strain/(1.0+4.0*strain)
    else:
        raise ValueError(mechanics_form)
    return mu_micro_n_m*1e-6 * b_m * (lam**2 - lam**-2) * factor


def residual(freq_hz: np.ndarray, dims_m: np.ndarray, a0: float,
             eps_mem: float, sigma_cyto: float, mu_micro_n_m: float,
             field_gain: float = 1.0, nonlinear_kappa: float = 0.0,
             sigma_mem: float = SIGMA_MEM,
             mechanics_form: str = "linear_plus") -> np.ndarray:
    dims = np.asarray(dims_m)
    lam = dims[..., 0]/a0
    if mechanics_form in {"qiang2018", "qiang2018_regularized",
                          "qiang2018_micromachines_regularized"}:
        # Qiang et al. Sci Rep 2018 define lambda_Q=b0/b, where b is the
        # minor semi-axis. For initially circular projected RBCs b0=a0;
        # diameter ratios are identical to semi-axis ratios. At equilibrium,
        # their Eq. 6-8 gives F=mu(lambda_Q)*b_diameter*(lambda_Q^2-lambda_Q^-2).
        lam_q = a0 / dims[..., 1]
        if mechanics_form == "qiang2018_micromachines_regularized":
            base_mu = 6.05
            high_branch = 0.28*np.exp(1.98*lam_q)+1.96
        else:
            base_mu = 3.02
            high_branch = 0.21*np.exp(2.06*lam_q)-0.25
        if mechanics_form == "qiang2018":
            mu_q = np.where(lam_q < 1.4, base_mu, high_branch)
        else:
            # The published piecewise fit jumps at 1.4, producing an interval
            # of DEP forces with no exact equilibrium root. Regularize only
            # lambda_Q in [1.38,1.42], with a fixed cubic smoothstep.
            x = np.clip((lam_q-1.38)/.04, 0, 1)
            blend = x*x*(3-2*x)
            mu_q = (1-blend)*base_mu + blend*high_branch
        # mu_micro_n_m/base_mu is an explicit cell-specific transfer factor;
        # the published relation itself is a population mean in another DEP setup.
        mech = (mu_micro_n_m/base_mu)*mu_q*1e-6*dims[..., 1]*(lam_q**2-lam_q**-2)
        return electrical_force(freq_hz,dims,eps_mem,sigma_cyto,field_gain,sigma_mem)-mech
    return (electrical_force(freq_hz, dims, eps_mem, sigma_cyto, field_gain, sigma_mem)
            - mechanical_force(lam, dims[..., 1], mu_micro_n_m,
                               nonlinear_kappa, mechanics_form))


def forward(freq_hz: np.ndarray, theta: np.ndarray, a0: float, b0: float, c0: float,
            closure: str = "balanced", field_gain: float = 1.0,
            nonlinear_kappa: float = 0.0, sigma_mem: float = SIGMA_MEM,
            mechanics_form: str = "linear_plus") -> np.ndarray:
    eps_mem, sigma_cyto, mu = theta
    def root_at(f):
        def balance(lam):
            dims = dimensions(lam, a0, b0, c0, closure)
            return float(residual(f, dims, a0, eps_mem, sigma_cyto, mu,
                                  field_gain, nonlinear_kappa, sigma_mem,
                                  mechanics_form))
        if balance(1.0) <= 0:
            return 1.0
        hi = 2.0
        while balance(hi) > 0 and hi < 8:
            hi *= 1.5
        if balance(hi) > 0:
            raise ValueError("No force-balance root below max stretch")
        return brentq(balance, 1.0, hi, xtol=1e-8)
    lam = np.array([root_at(f) for f in np.atleast_1d(freq_hz)])
    return dimensions(lam, a0, b0, c0, closure)


def infer(freq_hz: np.ndarray, measured_dims_m: np.ndarray, a0: float,
          start: tuple[float, float, float] = (4.7, .65, 2.5)) -> np.ndarray:
    """Inverse using the observed a,b and volume-derived c; field gain fixed 1."""
    dims = np.asarray(measured_dims_m)
    force_scale = 1e-11
    fit = least_squares(
        lambda p: residual(freq_hz, dims, a0, *np.exp(p)) / force_scale,
        np.log(start), bounds=(np.log([.1,.1,.1]), np.log([30,3,30])),
        max_nfev=80)
    return np.exp(fit.x)


def infer_nonlinear(freq_hz: np.ndarray, measured_dims_m: np.ndarray,
                    field_gain: np.ndarray, a0: float) -> np.ndarray:
    """Fit epsilon_mem, sigma_cyto, mu and nonlinear stiffness coefficient."""
    dims = np.asarray(measured_dims_m)
    fit = least_squares(
        lambda p: residual(freq_hz,dims,a0,*np.exp(p[:3]),
                           field_gain=field_gain,nonlinear_kappa=p[3])/1e-11,
        np.r_[np.log([4.7,.65,2.5]),1.0],
        bounds=(np.r_[np.log([.1,.1,.1]),0.0],
                np.r_[np.log([30,3,30]),5.0]),
        max_nfev=120)
    return np.r_[np.exp(fit.x[:3]),fit.x[3]]
