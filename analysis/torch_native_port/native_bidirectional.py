"""Differentiable 50-state endocardial ToR-ORd/SAC plus Land RHS.

The supported configuration uses the original SAC law. Only the Land beta
length sensitivity and an effective scale on the stretch-dependent electrical
RHS are exposed for exploratory virtual-cell heterogeneity. This scale is not
a molecular SAC conductance; most other cell-specific scales and alternative
feedback laws remain outside this interface.
"""
import torch
from .electrical_gating_partial import gating_intermediates
from .land2016_torch import land2016


def native_rhs(y, t=0.0, lam=1.0, trpn_on=True, sac_on=True,
               dlam_dt=0.0, sac_gain=None, sac_lam=None,
               land_beta=(2.3, -2.4)):
    y = torch.as_tensor(y)
    # When mechanics allows shortening below reference length, the published
    # SAC law is defined for tensile stretch only.  Keep Land's true length
    # while permitting its electrical input to be rectified to >= 1.
    ell = (lam if sac_lam is None else sac_lam) if sac_on else 1.0
    dy_e = gating_intermediates(y[..., :43], t=t, lam=ell,
                                return_intermediates=False)
    if sac_gain is not None and sac_on:
        resting = gating_intermediates(y[..., :43], t=t, lam=1.0,
                                      return_intermediates=False)
        dy_e = resting + sac_gain*(dy_e-resting)
    dy_l, T, Ta, Tp = land2016(y[..., 43:50], 1000*y[..., 5],
                              lam=lam, dlam_dt=dlam_dt, beta=land_beta)
    if trpn_on:
        cai = y[..., 5]
        cmdnmax, kmcmdn = .05, .00238
        trpnmax, kmtrpn = .07, .0005
        other = cmdnmax*kmcmdn/(kmcmdn+cai)**2
        static = trpnmax*kmtrpn/(kmtrpn+cai)**2
        B_old = 1/(1+other+static)
        B_new = 1/(1+other)
        cai_corr = B_new*(dy_e[..., 5]/B_old-trpnmax*dy_l[..., 2])
        dy_e = torch.cat([dy_e[..., :5], cai_corr.unsqueeze(-1), dy_e[..., 6:]], dim=-1)
    return torch.cat([dy_e,dy_l],dim=-1), (T,Ta,Tp)


def rk4_step(y, t, h, lam, trpn_on=True, sac_on=True, dlam_dt=0.0):
    k1,_=native_rhs(y,t,lam,trpn_on,sac_on,dlam_dt)
    k2,_=native_rhs(y+.5*h*k1,t+.5*h,lam,trpn_on,sac_on,dlam_dt)
    k3,_=native_rhs(y+.5*h*k2,t+.5*h,lam,trpn_on,sac_on,dlam_dt)
    k4,_=native_rhs(y+h*k3,t+h,lam,trpn_on,sac_on,dlam_dt)
    return y+(h/6)*(k1+2*k2+2*k3+k4)



