"""Symbolic checks for the minimal-measurement map of a force/tension/current twin.

All ranks are generic structural ranks for *rich noiseless input-output curves*.
They are upper bounds on what Duke-like sparse, noisy protocols can estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as s

HERE = Path(__file__).resolve().parent
F, T, T0, lam, T50, k, G, a, b = s.symbols(
    "F T T0 lambda T50 k G a b", real=True)
K, h, Tth, tau, N, gamma, E, V = s.symbols(
    "K h Tth tau N gamma E V", real=True)
tau_m, tau_g, laplace_s = s.symbols("tau_m tau_g s", positive=True)


def rank_of(functions, params):
    jac = s.Matrix(functions).jacobian(params)
    # Generic numerical substitution avoids expensive symbolic rank reduction.
    values = {T0: 1.3, lam: .007, T50: 2.5, k: .8, G: 200,
              K: 2.2, h: 2.3, Tth: 2.1, N: 23, gamma: 1.7, E: 0,
              V: -80, F: 50, tau: .02}
    return int(jac.subs(values).rank())


def check():
    # Sigmoid gate and its two-dimensional force-domain quotient.
    argument = (T0 + lam * F - T50) / k
    transformed = ((a*T0+b) + a*lam*F - (a*T50+b)) / (a*k)
    assert s.simplify(argument - transformed) == 0
    f50, sf = (T50-T0)/lam, k/lam
    assert s.simplify(argument-(F-f50)/sf) == 0
    sigmoid_params = [T0, lam, T50, k, G]
    ranks = {
        "force_current_sigmoid": rank_of([f50, sf, G], sigmoid_params),
        "plus_calibrated_delta_tension": rank_of([f50, sf, G, lam], sigmoid_params),
        "plus_absolute_tension": rank_of([f50, sf, G, lam, T0], sigmoid_params),
    }
    assert ranks == {"force_current_sigmoid": 3,
                     "plus_calibrated_delta_tension": 4,
                     "plus_absolute_tension": 5}

    # A hard threshold identifies only the force-domain crossing Fth.
    fth = (Tth-T0)/lam
    threshold_params = [T0, lam, Tth, G]
    threshold_rank = rank_of([fth, G], threshold_params)
    assert threshold_rank == 2
    assert s.simplify(((a*Tth+b)-(a*T0+b))/(a*lam)-fth) == 0

    # A Hill gate referenced to zero tension does not admit a translation
    # gauge, but the tension-unit scaling remains exact.
    hill = (T0+lam*F)**h / (K**h+(T0+lam*F)**h)
    scaled_hill = (a*T0+a*lam*F)**h / ((a*K)**h+(a*T0+a*lam*F)**h)
    # Symbolic power factoring requires positive bases. Check the algebraic
    # family explicitly and one nontrivial integer exponent substitution.
    assert s.simplify((a**h * (T0+lam*F)**h) /
                      (a**h*K**h + a**h*(T0+lam*F)**h) - hill) == 0
    assert s.simplify((scaled_hill-hill).subs({a: 2, h: 2})) == 0
    hill_combinations = [T0/lam, K/lam, h, G]
    hill_rank = rank_of(hill_combinations, [T0, lam, K, h, G])
    assert hill_rank == 4

    # For dT/dt=(T0+lambda*F-T)/tau, T'=aT+b obeys the same ODE form.
    rate = (T0+lam*F-T)/tau
    transformed_rate = ((a*T0+b)+a*lam*F-(a*T+b))/tau
    assert s.simplify(transformed_rate-a*rate) == 0

    # For two linear first-order stages in series, the output transfer
    # function identifies the two poles but not their mechanical/gating labels.
    cascade = 1/((1+tau_m*laplace_s)*(1+tau_g*laplace_s))
    assert s.simplify(cascade-cascade.subs({tau_m: tau_g, tau_g: tau_m},
                                           simultaneous=True)) == 0
    intermediate = 1/(1+tau_m*laplace_s)
    assert s.simplify(intermediate-intermediate.subs({tau_m: tau_g,
                                                      tau_g: tau_m},
                                                     simultaneous=True)) != 0

    # Unrestricted latent models admit nonlinear coordinate changes too.
    # Let z=T^2 for positive tension. dz/dt=2*T*dT/dt, and a readout of
    # sqrt(z) yields exactly the original current.
    t_positive = s.symbols("T_positive", positive=True)
    nonlinear_state = t_positive**2
    assert s.simplify(s.sqrt(nonlinear_state)-t_positive) == 0
    assert s.simplify(s.diff(nonlinear_state, t_positive)-2*t_positive) == 0

    # Voltage and single-channel amplitude: even a family of voltages only
    # identifies B=N*gamma and E if gating is shared across V.
    B = N*gamma
    voltage_rank = rank_of([B, E], [N, gamma, E])
    count_rank = rank_of([B, E, N], [N, gamma, E])
    assert voltage_rank == 2 and count_rank == 3

    result = {
        "assumptions": ["same cell and co-localized mechanical/electrical measurements",
                        "known gate family and sufficiently rich noiseless input curve",
                        "positive lambda and k; calibrated physical units for tension observations"],
        "sigmoid_static": {
            "model": "I=G sigmoid((T0+lambda F-T50)/k)",
            "gauge": "(T0,lambda,T50,k,G)->(aT0+b,a lambda,aT50+b,a k,G), a>0",
            "identifiable_force_domain": ["F50=(T50-T0)/lambda", "sF=k/lambda", "G"],
            "generic_rank_by_observation": ranks,
        },
        "hard_threshold": {
            "model": "I=G H(T0+lambda F-Tth)",
            "identifiable_force_domain": ["Fth=(Tth-T0)/lambda", "G"],
            "generic_rank_force_current": threshold_rank,
        },
        "hill": {
            "model": "I=G T^h/(K^h+T^h), T=T0+lambda F, T>0",
            "scale_gauge": "(T0,lambda,K)->a(T0,lambda,K)",
            "generic_force_domain_combinations": ["T0/lambda", "K/lambda", "h", "G"],
            "generic_rank_force_current": hill_rank,
            "qualification": "The zero of tension is fixed in this Hill form, so the sigmoid translation gauge does not apply. Structural rank does not guarantee practical estimability near a narrow response range.",
        },
        "latent_linear_mechanics": {
            "model": "dT/dt=(T0+lambda F-T)/tau_m; I=G sigmoid((T-T50)/k)",
            "affine_tension_gauge_survives": True,
            "additional_identifiable_candidate": "tau_m under time-resolved, sufficiently varied F(t), absent confounding electrical kinetics",
        },
        "two_stage_linear_dynamics": {
            "model": "tau_m dm/dt=F-m; tau_g dq/dt=m-q; I=Gq",
            "force_current_transfer": "G/((1+tau_m*s)(1+tau_g*s))",
            "exact_output_equivalence": "swap tau_m and tau_g",
            "intermediate_mechanics_transfer": "1/(1+tau_m*s); measuring m breaks label swap if sampling is adequate",
            "qualification": "This is a local linear-cascade example, not proof that every nonlinear Piezo model has the same swap ambiguity.",
        },
        "unrestricted_latent_dynamics": {
            "equivalence": "For any invertible state transform z=phi(T), use dz/dt=phi'(T)f(T,F) and readout h(phi^-1(z)); F-to-I is unchanged.",
            "example_checked": "z=T^2 for T>0",
            "implication": "A flexible NODE latent coordinate has no intrinsic tension unit or molecular interpretation without a measured intermediate or strong physical constraints.",
        },
        "adaptation": {
            "example": "dr/dt=(1-r)/tau_r-c p(T)r; I=G p(T)r(V-E)",
            "exact_tension_gauge_survives": True,
            "qualification": "Dynamics may identify recovery parameters only when the protocol varies p and recovery interval; one fixed pulse cannot establish this generically.",
        },
        "voltage_and_count": {
            "one_voltage": "G=N gamma (V-E) is one product",
            "multiple_voltages": "Can identify E and B=N gamma if gating and conductance are voltage-invariant over the tested range",
            "rank_voltage_only_for_N_gamma_E": voltage_rank,
            "rank_after_channel_count": count_rank,
        },
        "cross_dataset_rule": "Observations from different cell types, perturbations or mechanics geometries constrain priors; they do not break a same-cell gauge without justified parameter transport.",
    }
    (HERE / "mechanotransduction_observability_framework.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    check()
