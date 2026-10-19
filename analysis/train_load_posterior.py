"""Train/evaluate the load-conditioned posterior on frozen cell splits.

Usage: python analysis/train_load_posterior.py COHORT.h5 --epochs 120
The posterior observes only AP, Ca, T(lambda=1.0), T(lambda=1.1).
No outside-range load is ever passed to this inference network.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from load_conditioned_posterior import LoadConditionedPosterior


def read_data(path: Path):
    with h5py.File(path) as f:
        matlab = "cohort.mat" == path.name
        if matlab:
            v = f["V"][()].T
            ca = f["Cai_uM"][()].T
            tension = f["Ta_kPa"][()].transpose(2, 1, 0)
            theta = f["theta"][()].T
            split = f["split_code"][()].ravel()
            lam = f["lambda"][()].ravel()
            lo, hi = f["lo"][()].ravel(), f["hi"][()].ravel()
            ok = f["success"][()].ravel().astype(bool)
        else:
            v, ca = f["V"][()], f["Cai_uM"][()]
            tension, theta = f["Ta_kPa"][()], f["theta"][()]
            split = f["split_code"][()]
            lam, lo, hi = f["lambda"][()], f["lo"][()], f["hi"][()]
            ok = f["success"][()].astype(bool)
    if not ok.all():
        v, ca, tension, theta, split = (x[ok] for x in [v, ca, tension, theta, split])
    ix = [int(np.argmin(abs(lam - wanted))) for wanted in [1.0, 1.1]]
    if not np.allclose(lam[ix], [1.0, 1.1]):
        raise ValueError("Calibration loads missing")
    # 2-ms sampling keeps most of the upstroke while limiting CPU cost.
    return (
        torch.from_numpy(v[:, ::2].copy()).float(),
        torch.from_numpy(ca[:, ::2].copy()).float(),
        torch.from_numpy(tension[:, ix, ::2].copy()).float(),
        torch.tensor(lam[ix], dtype=torch.float32).repeat(len(v), 1),
        torch.from_numpy(((theta - lo) / (hi - lo)).copy()).float(),
        torch.from_numpy(split.astype(np.int64).copy()),
        lo, hi,
    )


def mode_masks(mode: int, batch: int, device):
    ca_mask = torch.full((batch,), float(mode >= 1), device=device)
    load_mask = torch.zeros(batch, 2, device=device)
    if mode >= 2:
        load_mask[:, 0] = 1
    if mode >= 3:
        load_mask[:, 1] = 1
    return ca_mask, load_mask


@torch.no_grad()
def validation_loss(model, dataset, indices, batch_size=128):
    model.eval()
    totals = []
    for mode in range(4):
        losses = []
        for start in range(0, len(indices), batch_size):
            ii = indices[start:start + batch_size]
            ap, ca, ta, lam, theta = (x[ii] for x in dataset[:5])
            cm, lm = mode_masks(mode, len(ii), ap.device)
            losses.extend((-model.log_prob(theta, model(ap, ca, ta, lam, cm, lm))).tolist())
        totals.append(float(np.mean(losses)))
    return float(np.mean(totals))


@torch.no_grad()
def evaluate(model, dataset, indices, n_draws=256):
    model.eval()
    ap, ca, ta, lam, truth = (x[indices] for x in dataset[:5])
    result = {}
    for mode in range(4):
        cm, lm = mode_masks(mode, len(indices), ap.device)
        posterior = model(ap, ca, ta, lam, cm, lm)
        draws = model.sample(posterior, n_draws)
        median = draws.median(dim=1).values
        low = torch.quantile(draws, .05, dim=1)
        high = torch.quantile(draws, .95, dim=1)
        cover = ((truth >= low) & (truth <= high)).float()
        result[mode] = {
            "nll": float((-model.log_prob(truth, posterior)).mean()),
            "electrical_median_MAE_prior_scaled": float((median[:, :4] - truth[:, :4]).abs().mean()),
            "mechanical_median_MAE_prior_scaled": float((median[:, 4:] - truth[:, 4:]).abs().mean()),
            "electrical_90pct_marginal_coverage": float(cover[:, :4].mean()),
            "mechanical_90pct_marginal_coverage": float(cover[:, 4:].mean()),
            "beta0_90pct_coverage": float(cover[:, 8].mean()),
            "beta1_90pct_coverage": float(cover[:, 9].mean()),
            "beta0_90pct_mean_width_prior_scaled": float((high[:, 8] - low[:, 8]).mean()),
            "beta1_90pct_mean_width_prior_scaled": float((high[:, 9] - low[:, 9]).mean()),
            "beta0_median_MAE_prior_scaled": float((median[:, 8] - truth[:, 8]).abs().mean()),
            "beta1_median_MAE_prior_scaled": float((median[:, 9] - truth[:, 9]).abs().mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    random.seed(20260928)
    dataset = read_data(args.cohort)
    split = dataset[5]
    train_idx = torch.where(split == 0)[0]
    valid_idx = torch.where(split == 1)[0]
    test_idx = torch.where(split == 2)[0]
    if min(map(len, [train_idx, valid_idx, test_idx])) < 10:
        raise ValueError("At least 10 cells per split required")
    model = LoadConditionedPosterior(n_times=dataset[0].shape[1])
    model.set_normalizers(dataset[0][train_idx], dataset[1][train_idx], dataset[2][train_idx])
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    batches = DataLoader(TensorDataset(*[x[train_idx] for x in dataset[:5]]),
                         batch_size=args.batch_size, shuffle=True)
    best = float("inf")
    patience = 20
    stale = 0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for ap, ca, ta, lam, theta in batches:
            mode = torch.randint(0, 4, (len(ap),))
            cm = (mode >= 1).float()
            lm = torch.stack(((mode >= 2).float(), (mode >= 3).float()), dim=1)
            optim.zero_grad(set_to_none=True)
            loss = -model.log_prob(theta, model(ap, ca, ta, lam, cm, lm)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.detach()))
        val = validation_loss(model, dataset, valid_idx)
        history.append({"epoch": epoch + 1, "train_nll": float(np.mean(losses)), "validation_nll": val})
        if val < best - 1e-4:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}: train={history[-1]['train_nll']:.3f} val={val:.3f} best={best:.3f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    result = {
        "cohort": str(args.cohort.resolve()), "train_n": len(train_idx),
        "validation_n": len(valid_idx), "test_n": len(test_idx),
        "best_validation_nll": best, "epochs_run": len(history),
        "test_modes": evaluate(model, dataset, test_idx),
        "mode_meanings": ["AP", "AP+Ca", "AP+Ca+T1.0", "AP+Ca+T1.0+T1.1"],
        "caution": "Synthetic in-distribution posterior only; no true-forward held-out-load prediction yet",
    }
    outdir = args.cohort.parent / (args.cohort.stem + "_posterior")
    outdir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "lo": dataset[6], "hi": dataset[7]}, outdir / "model.pt")
    (outdir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
