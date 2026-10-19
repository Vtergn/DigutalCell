"""Exact gauge-symmetry audit for the tension-gating layer.

This does not fit Duke data. It states which parameter combinations can be
identified from force/current when absolute local tension is unobserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
F, T0, lam, T50, k, G, a, b = sp.symbols(
    "F T0 lam T50 k G a b", real=True)


def gate(x):
    return 1/(1+sp.exp(-x))


original_argument = (T0+lam*F-T50)/k
transformed_argument = ((a*T0+b)+(a*lam)*F-(a*T50+b))/(a*k)
assert sp.simplify(transformed_argument-original_argument) == 0

F50 = (T50-T0)/lam
sF = k/lam
assert sp.simplify(original_argument-(F-F50)/sF) == 0

current = G*gate(original_argument)
assert sp.simplify(current-G*gate(transformed_argument)) == 0

# An observed, calibrated difference in tension fixes the multiplicative
# scale a, but cannot determine the additive offset b.
delta_tension = lam*F
delta_tension_transformed = a*lam*F
assert sp.simplify(delta_tension_transformed-delta_tension) == (a-1)*F*lam
absolute_tension = T0+lam*F
absolute_tension_transformed = a*T0+b+a*lam*F
assert sp.simplify(absolute_tension_transformed-absolute_tension-
                   ((a-1)*(T0+lam*F)+b)) == 0

result = {
    "model": "I(F)=G sigmoid((T0+lambda F-T50)/k)",
    "force_domain_combinations": ["F50_eff=(T50-T0)/lambda", "sF_eff=k/lambda", "G"],
    "gauge_transformation": {
        "T0": "a*T0+b", "lambda": "a*lambda", "T50": "a*T50+b",
        "k": "a*k", "G": "G", "condition": "a>0"
    },
    "exact_invariance_verified": True,
    "delta_tension_observation": "Breaks scale ambiguity a if calibrated in absolute units; leaves offset b.",
    "absolute_tension_observation": "Can anchor offset and scale if genuinely calibrated, spatially co-localized, and sampled across input conditions.",
    "one_shot": "One scalar peak cannot identify T0, lambda, and G simultaneously, even if T50 and k are fixed.",
    "qualification": "The structural symmetry applies when T50 and k are not fixed exactly for the same cell. With exact known T50,k and a rich force-current curve spanning threshold and saturation, T0,lambda,G can in principle be identified; Duke one-shot and near-similar stimuli remain practically weak."
}
(HERE/"observability_symmetry_audit.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
