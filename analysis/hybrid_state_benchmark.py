"""Exploratory two-state neural ODE baseline on the same held-out dates.

The 1-ms Euler update is a constrained latent-state system identification
benchmark. Its fitted time constants are not interpreted as Piezo kinetics.
"""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from causal_cnn_benchmark import HERE, build_records, metrics


SEED = 20260926
EPOCHS = 80
torch.set_num_threads(2)


class TwoStateNeuralODE(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.input = nn.Linear(inputs, 2)
        self.recurrent = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(inputs+2, 1)
        self.raw_tau = nn.Parameter(torch.tensor([-2.2, -0.85]))
        nn.init.constant_(self.readout.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dz/dt = [tanh(Ax+Bz)-z]/tau, integrated at 1 ms.
        state = x.new_zeros((x.shape[0], 2))
        inverse_tau = 1/(2+98*torch.sigmoid(self.raw_tau))
        output = []
        for index in range(x.shape[-1]):
            control = x[:, :, index]
            destination = torch.tanh(self.input(control)+self.recurrent(state))
            state = state+inverse_tau*(destination-state)
            response = F.softplus(self.readout(torch.cat((control, state), dim=1)))
            output.append(response[:, 0])
        return torch.stack(output, dim=1)

    def tau_ms(self) -> np.ndarray:
        return (2+98*torch.sigmoid(self.raw_tau)).detach().numpy()


def train(records: list[dict], heldout: str) -> tuple[TwoStateNeuralODE, float]:
    torch.manual_seed(SEED)
    model = TwoStateNeuralODE(records[0]["x"].shape[0])
    training = [r for r in records if r["date"] != heldout]
    x = torch.tensor(np.stack([r["x"] for r in training]))
    y = torch.tensor(np.stack([r["target"]/r["cal_peak"] for r in training]))
    counts = Counter(r["cell_id"] for r in training)
    cell_weight = torch.tensor([1/counts[r["cell_id"]] for r in training], dtype=torch.float32)
    cell_weight /= cell_weight.sum()
    stimulus_weight = 1+2*(y > .1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        prediction = model(x)
        loss = ((F.smooth_l1_loss(prediction, y, beta=.5, reduction="none")*
                 stimulus_weight).mean(dim=1)*cell_weight).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    return model, float(loss.detach())


def main() -> None:
    records = build_records()
    scores = []
    for date in sorted(set(r["date"] for r in records)):
        training = [r for r in records if r["date"] != date]
        test = [r for r in records if r["date"] == date]
        model, loss = train(records, date)
        model.eval()
        with torch.no_grad():
            predicted = model(torch.tensor(np.stack([r["x"] for r in test]))).numpy()
        tau = model.tau_ms()
        for row, normalized in zip(test, predicted):
            waveform = normalized*row["cal_peak"]
            scores.append({"cell_id": row["cell_id"], "channel": row["channel"],
                           "heldout_date": date, "sweep": row["sweep"],
                           "model": "two_state_neural_ode", "train_cells": len(set(r["cell_id"] for r in training)),
                           "train_sweeps": len(training), "train_loss_normalized": loss,
                           "latent_tau1_ms": float(tau[0]), "latent_tau2_ms": float(tau[1]),
                           **metrics(row, waveform)})
        print("fold", date, "tau", tau.round(1), "loss", round(loss, 3), flush=True)
    with (HERE / "hybrid_state_benchmark_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    print("saved", len(scores), "scores", "parameters",
          sum(parameter.numel() for parameter in TwoStateNeuralODE(records[0]["x"].shape[0]).parameters()))


if __name__ == "__main__":
    main()
