"""Load-conditioned posterior network for the paired virtual-cell cohort.

This file defines and smoke-tests the architecture. No trained weights or
posterior accuracy claims are produced by its self-test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


class TraceEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(32 * 8, 48), nn.GELU(),
        )

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        return self.net(trace)


class LoadConditionedPosterior(nn.Module):
    """Permutation-invariant tension-load set, 3-component logistic-normal q(theta|D)."""

    def __init__(self, n_params: int = 12, n_mixtures: int = 3, n_times: int = 1) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_mixtures = n_mixtures
        self.ap = TraceEncoder(1)
        self.ca = TraceEncoder(1)
        self.tension = TraceEncoder(2)
        self.register_buffer("ap_mean", torch.zeros(1, n_times))
        self.register_buffer("ap_std", torch.ones(1, n_times))
        self.register_buffer("ca_mean", torch.zeros(1, n_times))
        self.register_buffer("ca_std", torch.ones(1, n_times))
        self.register_buffer("t_mean", torch.zeros(1, 1, n_times))
        self.register_buffer("t_std", torch.ones(1, 1, n_times))
        self.head = nn.Sequential(
            nn.Linear(48 * 3 + 3, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, n_mixtures * (2 * n_params + 1)),
        )

    def set_normalizers(self, ap: torch.Tensor, ca: torch.Tensor, t: torch.Tensor) -> None:
        """Fit per-time normalization on training cells only."""
        self.ap_mean = ap.mean(0, keepdim=True)
        self.ap_std = ap.std(0, keepdim=True).clamp_min(1.0)
        self.ca_mean = ca.mean(0, keepdim=True)
        self.ca_std = ca.std(0, keepdim=True).clamp_min(0.01)
        self.t_mean = t.mean((0, 1), keepdim=True)
        self.t_std = t.std((0, 1), keepdim=True).clamp_min(1.0)

    def forward(
        self,
        ap: torch.Tensor,
        ca: torch.Tensor,
        tension: torch.Tensor,
        lambdas: torch.Tensor,
        ca_mask: torch.Tensor,
        load_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AP,Ca [B,L]; tension [B,K,L]; lambdas/load_mask [B,K]."""
        b, k, length = tension.shape
        if ap.shape != (b, length) or ca.shape != (b, length):
            raise ValueError("AP, Ca and tension must share batch and time axes")
        if lambdas.shape != (b, k) or load_mask.shape != (b, k):
            raise ValueError("Invalid load set shape")
        ap_z = self.ap(((ap - self.ap_mean) / self.ap_std).unsqueeze(1))
        ca_z = self.ca(((ca - self.ca_mean) / self.ca_std).unsqueeze(1)) * ca_mask[:, None]
        normalized_lambda = ((lambdas - 1.0) / 0.125)[:, :, None].expand(-1, -1, length)
        mechanics = torch.stack(((tension - self.t_mean) / self.t_std, normalized_lambda), dim=2)
        mech_z = self.tension(mechanics.reshape(b * k, 2, length)).reshape(b, k, 48)
        mech_z = (mech_z * load_mask[:, :, None]).sum(1) / load_mask.sum(1, keepdim=True).clamp_min(1)
        flags = torch.stack((ca_mask, (load_mask.sum(1) > 0).float(), load_mask.sum(1) / k), dim=1)
        output = self.head(torch.cat((ap_z, ca_z, mech_z, flags), dim=1))
        output = output.reshape(b, self.n_mixtures, 2 * self.n_params + 1)
        logits = output[:, :, 0]
        mean = output[:, :, 1:1 + self.n_params]
        log_sd = output[:, :, 1 + self.n_params:].clamp(-4, 2)
        return logits, mean, log_sd

    @staticmethod
    def log_prob(
        theta_unit: torch.Tensor,
        posterior: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Density for theta scaled into the open unit hypercube."""
        logits, mean, log_sd = posterior
        u = theta_unit.clamp(1e-5, 1 - 1e-5)
        z = torch.logit(u)[:, None, :]
        log_normal = (-0.5 * ((z - mean) / log_sd.exp()).square()
                      - log_sd - 0.5 * np.log(2 * np.pi)).sum(-1)
        jac = (-torch.log(u) - torch.log1p(-u)).sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, 1) + log_normal, dim=1) + jac

    @staticmethod
    def sample(
        posterior: tuple[torch.Tensor, torch.Tensor, torch.Tensor], n: int
    ) -> torch.Tensor:
        logits, mean, log_sd = posterior
        b, _, d = mean.shape
        component = torch.distributions.Categorical(logits=logits).sample((n,)).T
        chosen_mean = mean.gather(1, component[:, :, None].expand(b, n, d))
        chosen_sd = log_sd.exp().gather(1, component[:, :, None].expand(b, n, d))
        return torch.sigmoid(chosen_mean + chosen_sd * torch.randn_like(chosen_mean))


def smoke(path: Path) -> None:
    with h5py.File(path / "cohort.mat") as f:
        ok = np.flatnonzero(f["success"][()].ravel().astype(bool))[:4]
        v = torch.tensor(f["V"][()].T[ok, ::5], dtype=torch.float32)
        ca = torch.tensor(f["Cai_uM"][()].T[ok, ::5], dtype=torch.float32)
        ta = f["Ta_kPa"][()].transpose(2, 1, 0)[ok, :, ::5]
        loads = f["lambda"][()].ravel()
        ix = [int(np.argmin(abs(loads - x))) for x in [1.0, 1.1]]
        tension = torch.tensor(ta[:, ix, :], dtype=torch.float32)
        lam = torch.tensor(loads[ix], dtype=torch.float32).repeat(len(ok), 1)
        lo, hi = f["lo"][()].ravel(), f["hi"][()].ravel()
        target = torch.tensor((f["theta"][()].T[ok] - lo) / (hi - lo), dtype=torch.float32)
    model = LoadConditionedPosterior()
    masks = torch.tensor([[1., 0.], [1., 1.], [0., 0.], [1., 1.]])
    posterior = model(v, ca, tension, lam, torch.tensor([1., 1., 0., 1.]), masks)
    reordered = model(v, ca, tension.flip(1), lam.flip(1),
                      torch.tensor([1., 1., 0., 1.]), masks.flip(1))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(posterior, reordered))
    log_prob = model.log_prob(target, posterior)
    draws = model.sample(posterior, 5)
    assert log_prob.shape == (len(ok),) and torch.isfinite(log_prob).all()
    assert draws.shape == (len(ok), 5, 12)
    assert torch.all((draws > 0) & (draws < 1))
    print(f"set encoder/posterior smoke passed: {len(ok)} cells, 0/1/2 loads")


if __name__ == "__main__":
    smoke(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "virtual_cardiomyocyte_96_v2")
