"""PyTorch differentiable implementation of the intact Land 2016 model.

This mirrors Data/7fb/.../Land2016.m for scalar Cai and scalar length input.
The piecewise branches use torch.where so gradients are defined almost
everywhere (the original model itself is piecewise at branch boundaries).
"""
import torch


def land2016(y, cai_uM, lam=1.0, dlam_dt=0.0, beta=(2.3, -2.4),
             kinetics=None):
    y = torch.as_tensor(y)
    dtype, dev = y.dtype, y.device
    lam = torch.as_tensor(lam, dtype=dtype, device=dev)
    dlam_dt = torch.as_tensor(dlam_dt, dtype=dtype, device=dev)
    cai = torch.as_tensor(cai_uM, dtype=dtype, device=dev)
    beta0 = torch.as_tensor(beta[0], dtype=dtype, device=dev)
    beta1 = torch.as_tensor(beta[1], dtype=dtype, device=dev)

    perm50, trpn_n, koff, dr, wfrac = 0.35, 2.0, 0.1, 0.25, 0.5
    TOT_A, ktm_unblock = 25.0, 1.0
    gamma, gamma_wu, phi = 0.0085, 0.615, 2.23
    nperm, ca50, Tref, nu, mu = 5.0, 0.805, 120.0, 7.0, 3.0
    if kinetics is not None:
        # Optional reference-model comparison. The default still mirrors
        # Land2016.m's intact branch exactly.
        ktm_unblock = kinetics.get("ktm_unblock", ktm_unblock)
        nperm = kinetics.get("nperm", nperm)

    XS, XW, TRPN, TmBlocked, ZETAS, ZETAW, Cd = [y[..., i] for i in range(7)]
    XS = torch.clamp(XS, min=0); XW = torch.clamp(XW, min=0); TRPN = torch.clamp(TRPN, min=0)
    lam_c = torch.clamp(lam, max=1.2)
    min_lam = torch.minimum(torch.full_like(lam_c, 0.87), lam_c)
    Lfac = torch.clamp(1 + beta0 * (lam_c + min_lam - 1.87), min=0)
    XU = (1 - TmBlocked) - XW - XS

    k_ws, k_uw = 0.004 * mu, 0.026 * nu
    cdw = phi * k_uw * (1-dr) * (1-wfrac) / ((1-dr) * wfrac)
    cds = phi * k_ws * (1-dr) * wfrac / dr
    k_wu = k_uw * (1/wfrac - 1) - k_ws
    k_su = k_ws * (1/dr - 1) * wfrac
    A = (0.25 * TOT_A) / ((1-dr) * wfrac + dr) * (dr / 0.25)

    xb_ws = k_ws * XW
    xb_uw = k_uw * XU
    xb_wu = k_wu * XW
    xb_su = k_su * XS
    gamma_rate =  gamma * torch.maximum(torch.clamp(ZETAS, min=0), torch.clamp(-ZETAS-1, min=0))
    gamma_rate_w = gamma_wu * torch.abs(ZETAW)
    dXS = xb_ws - xb_su - gamma_rate * XS
    dXW = xb_uw - xb_wu - xb_ws - gamma_rate_w * XW

    ca50_eff = ca50 + beta1 * torch.minimum(torch.full_like(lam_c, 0.2), lam_c - 1)
    dTRPN = koff * ((cai / ca50_eff).clamp(min=0) ** trpn_n * (1 - TRPN) - TRPN)
    XSSS, XWSS = dr * 0.5, (1-dr) * wfrac * 0.5
    ktm_block = ktm_unblock * (perm50 ** nperm) * 0.5 / (0.5 - XSSS - XWSS)
    trpn_safe = torch.clamp(TRPN, min=1e-12)
    dTm = ktm_block * torch.minimum(torch.full_like(TRPN, 100.), trpn_safe ** (-nperm/2)) * XU \
        - ktm_unblock * (trpn_safe ** (nperm/2)) * TmBlocked
    dZs = A * dlam_dt - cds * ZETAS
    dZw = A * dlam_dt - (phi * k_uw * (1-dr) * (1-wfrac) / ((1-dr)*wfrac)) * ZETAW

    C = lam - 1
    eta = torch.where((C - Cd) < 0, torch.full_like(C, 20.), torch.full_like(C, 200.))
    dCd = 7.0 * (C - Cd) / eta
    Fd = eta * dCd
    Tp = 2.1 * (torch.exp(9.1 * C) - 1 + Fd)
    Ta = Lfac * (Tref / dr) * ((ZETAS + 1) * XS + ZETAW * XW)
    T = Ta + Tp
    dy = torch.stack([dXS, dXW, dTRPN, dTm, dZs, dZw, dCd], dim=-1)
    return dy, T, Ta, Tp

