"""Repeat-held-out ETH activation benchmark and independent eLife prior audit.

Each ETH row is one cell threshold. An experimental repeat, not an expanded
pressure-grid row, is the held-out unit. The grid simply turns a threshold
into a response-by-pressure curve for scoring.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data"
PRESSURE_GRID = np.arange(0, 400, 25, dtype=float)  # 400 excluded: possible endpoint censoring
MAIN_FORCES = {"Control": {2, 12.5, 25, 37.5, 50},
               "MargAc": {2, 6, 12.5, 18, 25},
               "CytoD": {2, 6, 12.5, 18, 25}}


def workbook_values(path: Path, sheet: str) -> list[tuple]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(workbook[sheet].values)
    finally:
        workbook.close()


def eth_rows() -> list[dict]:
    path = next((DATA / "CalciumImaging").rglob("SourceData_*.xlsx"))
    rows = workbook_values(path, "Critical Aspiration Pressures")[1:]
    return [{"condition": str(row[0]), "repeat": int(row[1]),
             "force_nN": float(row[2]), "threshold_mbar": float(row[3])}
            for row in rows if row[0] in MAIN_FORCES and row[2] in MAIN_FORCES[row[0]]]


def design(pressure: np.ndarray, force: np.ndarray, model: str) -> np.ndarray:
    parts = [np.ones_like(pressure), pressure/100]
    if model in {"additive", "quadratic"}:
        parts.append(force/20)
    if model == "quadratic":
        parts.append((force/20)**2)
    return np.stack(parts, axis=-1)


def prepare(rows: list[dict], model: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.broadcast_to(PRESSURE_GRID[None, :], (len(rows), len(PRESSURE_GRID)))
    f = np.array([row["force_nN"] for row in rows])[:, None]
    f = np.broadcast_to(f, p.shape)
    threshold = np.array([row["threshold_mbar"] for row in rows])[:, None]
    target = (p >= threshold).astype(float)
    return design(p, f, model).reshape(-1, design(p, f, model).shape[-1]), target.ravel()


def fit(rows: list[dict], model: str) -> np.ndarray:
    matrix, target = prepare(rows, model)
    penalty = .001

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logit = matrix@coef
        probability = expit(logit)
        value = np.mean(np.logaddexp(0, logit)-target*logit)+penalty*np.sum(coef[1:]**2)
        gradient = matrix.T@(probability-target)/len(target)
        gradient[1:] += 2*penalty*coef[1:]
        return float(value), gradient

    result = minimize(objective, np.zeros(matrix.shape[1]), method="L-BFGS-B", jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def score(rows: list[dict], model: str, coefficient: np.ndarray) -> dict[str, float]:
    matrix, target = prepare(rows, model)
    probability = np.clip(expit(matrix@coefficient), 1e-9, 1-1e-9)
    return {"brier": float(np.mean((probability-target)**2)),
            "logloss": float(np.mean(-target*np.log(probability)-(1-target)*np.log1p(-probability)))}


def force_equivalent(coefficient: np.ndarray) -> float:
    return float(5*coefficient[2]/coefficient[1])


def elife_prior() -> dict:
    path = DATA / "elife-70988-fig5-data1-v1.xlsx"
    t50_rows = [r for r in workbook_values(path, "Figure 5D")[1:]
                if isinstance(r[0], (float, int)) and isinstance(r[1], (float, int))]
    slope_rows = [r for r in workbook_values(path, "Figure 5E")[1:]
                  if isinstance(r[0], (float, int)) and isinstance(r[1], (float, int))]
    if len(t50_rows) != len(slope_rows) or any(abs(a[0]-b[0]) > 1e-10
                                                for a, b in zip(t50_rows, slope_rows)):
        raise ValueError("Figure 5D/5E density rows are not aligned")
    density = np.array([r[0] for r in t50_rows])
    t50 = np.array([r[1] for r in t50_rows])
    slope = np.array([r[1] for r in slope_rows])
    return {"patches": len(t50_rows),
            "t50_mN_per_m": {"median": float(np.median(t50)),
                              "q25": float(np.quantile(t50, .25)),
                              "q75": float(np.quantile(t50, .75)),
                              "range": [float(t50.min()), float(t50.max())]},
            "slope_mN_per_m": {"median": float(np.median(slope)),
                                "q25": float(np.quantile(slope, .25)),
                                "q75": float(np.quantile(slope, .75)),
                                "range": [float(slope.min()), float(slope.max())]},
            "density_spearman_t50": float(spearmanr(density, t50).statistic),
            "density_spearman_t50_p": float(spearmanr(density, t50).pvalue),
            "density_spearman_slope": float(spearmanr(density, slope).statistic),
            "density_spearman_slope_p": float(spearmanr(density, slope).pvalue)}


def main() -> None:
    rows = eth_rows()
    report: dict = {"eth_main_cells": len(rows), "eth_by_condition": dict(Counter(r["condition"] for r in rows)),
                    "pressure_grid_mbar": PRESSURE_GRID.tolist(), "eth_models": {},
                    "elife": elife_prior()}
    eth_path = next((DATA / "CalciumImaging").rglob("SourceData_*.xlsx"))
    fit_points = workbook_values(eth_path, "pC,50 per F")[1:]
    for condition in MAIN_FORCES:
        group = [row for row in rows if row["condition"] == condition]
        heldout = []
        for repeat in (1, 2, 3):
            train = [row for row in group if row["repeat"] != repeat]
            test = [row for row in group if row["repeat"] == repeat]
            models = {name: fit(train, name) for name in ("pressure_only", "additive", "quadratic")}
            heldout.append({"repeat": repeat, "train_cells": len(train), "test_cells": len(test),
                            "scores": {name: score(test, name, coef) for name, coef in models.items()},
                            "force_equivalent_mbar_per_nN": force_equivalent(models["additive"])})
        full_additive = fit(group, "additive")
        published_points = [(float(r[1]), float(r[2])) for r in fit_points if r[0] == condition]
        slope, intercept = np.polyfit([p[0] for p in published_points],
                                      [p[1] for p in published_points], 1)
        report["eth_models"][condition] = {
            "cells": len(group), "repeats": dict(Counter(r["repeat"] for r in group)),
            "heldout": heldout,
            "full_additive_force_equivalent_mbar_per_nN": force_equivalent(full_additive),
            "source_p50_slope_mbar_per_nN": float(slope),
            "source_p50_intercept_mbar": float(intercept),
            "source_p50_points": published_points,
        }
        print(condition, "cells", len(group), "source p50 slope", round(float(slope), 2),
              "additive beta", round(force_equivalent(full_additive), 2))
        for row in heldout:
            print(" repeat", row["repeat"], "test", row["test_cells"],
                  "Brier", {k: round(v["brier"], 3) for k, v in row["scores"].items()},
                  "beta", round(row["force_equivalent_mbar_per_nN"], 2))
    print("eLife", report["elife"])
    output = HERE / "external_physics_bridge_results.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", output)


if __name__ == "__main__":
    main()
