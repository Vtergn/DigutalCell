"""Falsification pilot for sparse-intervention waveform factors.

Synthetic interventions are paired within a simulated cell. Real Duke cells are
not paired across interventions, so sparse intervention supervision is *not*
applied to real data. Real evaluation is only held-date shape autoencoding.
Observed current is provided to the encoder; no result is forward prediction.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SEED = 20260928
torch.set_num_threads(1)


class Autoencoder(torch.nn.Module):
    def __init__(self, width: int, latent: int = 4):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(width, 48), torch.nn.Tanh(),
                                           torch.nn.Linear(48, latent))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(latent, 48), torch.nn.Tanh(),
                                           torch.nn.Linear(48, width))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class ConditionalAE(torch.nn.Module):
    def __init__(self, current_width: int, mechanical_width: int, latent: int = 4):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(current_width+mechanical_width, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, latent))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent+mechanical_width, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, current_width))

    def forward(self, current: torch.Tensor,
                mechanics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(torch.cat([current, mechanics], dim=1))
        return self.decoder(torch.cat([z, mechanics], dim=1)), z


def synthetic(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows grouped [control, threshold, delay, decay] for each base cell."""
    rng = np.random.default_rng(seed)
    time = np.arange(201, dtype=float)
    curves, truth, mechanics = [], [], []
    for _ in range(180):
        force_max = rng.uniform(280, 540)
        base_threshold = rng.uniform(80, 200)
        base_delay = rng.uniform(1.5, 13)
        base_tau = rng.uniform(13, 48)
        gain = rng.lognormal(0, .35)
        rise_time = rng.uniform(37, 55)
        force = force_max*np.minimum(time/rise_time, 1)
        force *= np.where(time < rise_time, 1.,
                          .78+.22*np.exp(-(time-rise_time)/65))
        for arm in range(4):
            threshold = base_threshold*(.65 if arm == 1 else 1.)
            delay = max(.1, base_delay-(5 if arm == 2 else 0))
            tau = base_tau*(.55 if arm == 3 else 1.)
            crossing = int(np.flatnonzero(force > threshold)[0])
            elapsed = np.maximum(time-crossing-delay, 0)
            response = (1-np.exp(-elapsed/2.7))*np.exp(-elapsed/tau)
            response *= gain*((force_max-threshold)/250)**.7
            response += rng.normal(0, .012, size=len(time))
            curves.append(response)
            truth.append([threshold, delay, tau, gain])
            work = np.cumsum(force)
            mechanics.append(np.concatenate([force/500, work/np.max(work)]))
    return (np.asarray(curves, dtype=np.float32), np.asarray(truth),
            np.asarray(mechanics, dtype=np.float32))


def intervention_penalty(z: torch.Tensor) -> torch.Tensor:
    delta = z.reshape(-1, 4, 4)[:, 1:] - z.reshape(-1, 4, 4)[:, :1]
    shifts = torch.mean(delta, dim=0)
    concentration = (torch.linalg.vector_norm(shifts, ord=1, dim=1)
                     - torch.linalg.vector_norm(shifts, ord=2, dim=1)).mean()
    unit = shifts/(torch.linalg.vector_norm(shifts, dim=1, keepdim=True)+1e-6)
    gram = unit@unit.T
    off_diagonal = (gram-torch.diag_embed(torch.diagonal(gram))).square().sum()/6
    shift_floor = torch.relu(.15-torch.linalg.vector_norm(shifts, dim=1)).square().mean()
    return .02*concentration+.05*off_diagonal+.01*shift_floor


def fit_ae(train: np.ndarray, *, paired: bool, seed: int, epochs: int = 350) -> Autoencoder:
    torch.manual_seed(seed)
    model = Autoencoder(train.shape[1])
    x = torch.tensor(train, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.003)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed, z = model(x)
        loss = torch.mean((reconstructed-x)**2)
        if paired:
            loss = loss+intervention_penalty(z)
        loss.backward()
        optimizer.step()
    return model


def transform(model: Autoencoder, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        reconstructed, z = model(torch.tensor(x, dtype=torch.float32))
    return reconstructed.numpy(), z.numpy()


def fit_conditional(train: np.ndarray, mechanical: np.ndarray, *, paired: bool,
                    seed: int, epochs: int = 350) -> ConditionalAE:
    torch.manual_seed(seed)
    model = ConditionalAE(train.shape[1], mechanical.shape[1])
    y = torch.tensor(train, dtype=torch.float32)
    m = torch.tensor(mechanical, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.003)
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, z = model(y, m)
        loss = torch.mean((reconstructed-y)**2)
        if paired:
            loss = loss+intervention_penalty(z)
        loss.backward()
        optimizer.step()
    return model


def transform_conditional(model: ConditionalAE, y: np.ndarray,
                          mechanical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred, z = model(torch.tensor(y, dtype=torch.float32),
                        torch.tensor(mechanical, dtype=torch.float32))
    return pred.numpy(), z.numpy()


def conditional_pca(train: np.ndarray, mechanical_train: np.ndarray,
                    test: np.ndarray, mechanical_test: np.ndarray,
                    rank: int = 4) -> tuple[np.ndarray, np.ndarray]:
    width = train.shape[1]
    # Three pointwise coefficients per time sample: intercept, force, work.
    m_train = mechanical_train.reshape(-1, 2, width)
    m_test = mechanical_test.reshape(-1, 2, width)
    base_train = np.stack([np.ones_like(train), m_train[:, 0], m_train[:, 1]], axis=-1)
    base_test = np.stack([np.ones_like(test), m_test[:, 0], m_test[:, 1]], axis=-1)
    gram = np.einsum('ntp,ntq->tpq', base_train, base_train)
    right = np.einsum('ntp,nt->tp', base_train, train)
    gram += np.eye(3, dtype=np.float32)[None, :, :]*.01
    coef = np.linalg.solve(gram, right[..., None])[..., 0]
    baseline_train = np.einsum('ntp,tp->nt', base_train, coef)
    baseline_test = np.einsum('ntp,tp->nt', base_test, coef)
    residual_pred, z = pca(train-baseline_train, test-baseline_test, rank)
    return baseline_test+residual_pred, z


def pca(train: np.ndarray, test: np.ndarray, rank: int = 4) -> tuple[np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train-center, full_matrices=False)
    axes = vt[:rank]
    z = (test-center)@axes.T
    return z@axes+center, z


def factor_alignment(z: np.ndarray, truth: np.ndarray) -> float:
    # Match factors after training; only signs and permutations are allowed.
    corr = np.asarray([[abs(float(spearmanr(z[:, i], truth[:, j]).statistic))
                        for j in range(truth.shape[1])] for i in range(z.shape[1])])
    rows, cols = linear_sum_assignment(-corr)
    return float(np.mean(corr[rows, cols]))


def shift_concentration(z: np.ndarray) -> dict:
    delta = z.reshape(-1, 4, 4)[:, 1:] - z.reshape(-1, 4, 4)[:, :1]
    means = delta.mean(axis=0)
    dominant = np.argmax(abs(means), axis=1)
    return {"mean_dominant_fraction": float(np.mean(np.max(abs(means), axis=1) /
                                                  np.maximum(np.sum(abs(means), axis=1), 1e-9))),
            "dominant_latent_indices_by_intervention": dominant.tolist(),
            "distinct_axes_used": int(len(set(dominant.tolist())))}


def synthetic_benchmark() -> dict:
    shapes, truth, mechanics = synthetic()
    rng = np.random.default_rng(SEED)
    bases = rng.permutation(len(shapes)//4)
    train_base, test_base = bases[:120], bases[120:]
    train_idx = (train_base[:, None]*4+np.arange(4)).ravel()
    test_idx = (test_base[:, None]*4+np.arange(4)).ravel()
    train, test = shapes[train_idx], shapes[test_idx]
    mechanical_train, mechanical_test = mechanics[train_idx], mechanics[test_idx]
    # Fixed overall scaling; no held-out data used to choose it.
    scale = max(float(np.std(train)), 1e-6)
    train, test = train/scale, test/scale
    output = {}
    for name in ("PCA4", "plain_AE4", "paired_sparse_orth_AE4",
                 "conditional_PCA4", "conditional_AE4", "conditional_sparse_orth_AE4"):
        if name == "PCA4":
            pred, z = pca(train, test)
        elif name == "conditional_PCA4":
            pred, z = conditional_pca(train, mechanical_train, test, mechanical_test)
        elif name.startswith("conditional_"):
            model = fit_conditional(train, mechanical_train,
                                    paired=name == "conditional_sparse_orth_AE4", seed=SEED)
            pred, z = transform_conditional(model, test, mechanical_test)
        else:
            model = fit_ae(train, paired=name == "paired_sparse_orth_AE4", seed=SEED)
            pred, z = transform(model, test)
        output[name] = {"held_base_cells": len(test_base),
                        "shape_rmse": float(np.sqrt(np.mean((pred-test)**2))),
                        "factor_alignment_mean_abs_spearman":
                            factor_alignment(z, truth[test_idx]),
                        **shift_concentration(z)}
    return output


def synthetic_seed_check() -> list[dict]:
    shapes, truth, mechanics = synthetic()
    rng = np.random.default_rng(SEED)
    bases = rng.permutation(len(shapes)//4)
    train_idx = (bases[:120, None]*4+np.arange(4)).ravel()
    test_idx = (bases[120:, None]*4+np.arange(4)).ravel()
    scale = max(float(np.std(shapes[train_idx])), 1e-6)
    train, test = shapes[train_idx]/scale, shapes[test_idx]/scale
    output = []
    for seed in (SEED, SEED+1, SEED+2):
        for paired in (False, True):
            model = fit_conditional(train, mechanics[train_idx], paired=paired,
                                    seed=seed)
            pred, z = transform_conditional(model, test, mechanics[test_idx])
            output.append({"seed": seed, "intervention_constraint": int(paired),
                           "shape_rmse": float(np.sqrt(np.mean((pred-test)**2))),
                           "factor_alignment_mean_abs_spearman":
                               factor_alignment(z, truth[test_idx]),
                           **shift_concentration(z)})
    return output


def real_benchmark() -> tuple[dict, list[dict]]:
    data = np.load(HERE / "frozen_wt_waveform_arrays.npz")
    shape = data["current_shape"].astype(np.float32)
    mechanical = data["mechanical"].astype(np.float32)
    mechanical[:, 0] /= 500
    mechanical[:, 1] /= 500
    mechanical = mechanical.reshape(len(mechanical), -1)
    groups = data["group"].astype(str)
    cells = data["cell_id"].astype(str)
    dates = data["date"].astype(str)
    wt = np.isin(groups, ["WT_mp1", "WT_mp2"])
    held_rows = []
    for date in sorted(set(dates[wt])):
        train_idx = np.flatnonzero(wt & (dates != date))
        test_idx = np.flatnonzero(wt & (dates == date))
        train, test = shape[train_idx], shape[test_idx]
        mechanical_train, mechanical_test = mechanical[train_idx], mechanical[test_idx]
        for model_name in ("PCA4", "plain_AE4", "conditional_PCA4",
                           "conditional_AE4"):
            if model_name == "PCA4":
                pred, _ = pca(train, test)
            elif model_name == "conditional_PCA4":
                pred, _ = conditional_pca(train, mechanical_train, test,
                                          mechanical_test)
            elif model_name == "conditional_AE4":
                model = fit_conditional(train, mechanical_train, paired=False,
                                        seed=SEED, epochs=250)
                pred, _ = transform_conditional(model, test, mechanical_test)
            else:
                model = fit_ae(train, paired=False, seed=SEED, epochs=250)
                pred, _ = transform(model, test)
            for i, j in enumerate(test_idx):
                held_rows.append({"model": model_name, "date": date,
                                  "cell_id": cells[j], "group": groups[j],
                                  "sweep": int(data["sweep"][j]),
                                  "shape_rmse": float(np.sqrt(np.mean((pred[i]-test[i])**2)))})
    # Collapse sweeps within each cell; cells within dates, then dates within
    # channel. A date is never represented by dozens of pseudo-independent sweeps.
    metrics = {}
    for model_name in ("PCA4", "plain_AE4", "conditional_PCA4",
                       "conditional_AE4"):
        rows = [r for r in held_rows if r["model"] == model_name]
        cell = defaultdict(list)
        for row in rows:
            cell[(row["group"], row["date"], row["cell_id"])].append(row["shape_rmse"])
        date = defaultdict(list)
        for (group, day, _), values in cell.items():
            date[(group, day)].append(float(np.mean(values)))
        metrics[model_name] = {group: {
            "dates": len([1 for g, _ in date if g == group]),
            "cells": len([1 for g, _, _ in cell if g == group]),
            "mean_of_date_mean_cell_rmse": float(np.mean([
                np.mean(values) for (g, _), values in date.items() if g == group]))}
            for group in ("WT_mp1", "WT_mp2")}
    return metrics, held_rows


def main() -> None:
    synthetic_results = synthetic_benchmark()
    seed_check = synthetic_seed_check()
    real_results, held_rows = real_benchmark()
    result = {"synthetic": synthetic_results,
              "conditional_synthetic_three_seed_check": seed_check,
              "real_WT_leave_whole_date_out": real_results,
              "limits": "Synthetic pairs exist by construction; Duke perturbations are unpaired, CytoD is confounded with cell-number block, and WT channel is confounded with date. The autoencoder sees observed current, so held-date reconstruction does not show future prediction or biological factor identification."}
    (HERE / "factorized_latent_feasibility_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (HERE / "factorized_latent_real_held_date.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(held_rows[0]))
        writer.writeheader()
        writer.writerows(held_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
