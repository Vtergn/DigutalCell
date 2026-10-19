"""Freeze off-grid three-parameter MCT development cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
LIMITS = np.array([[0.20, 1.30], [0.90, 1.10], [0.85, 1.15]])


def sample(n: int, seed: int, split: str) -> list[dict]:
    unit = qmc.LatinHypercube(d=3, seed=seed).random(n)
    values = qmc.scale(unit, LIMITS[:, 0], LIMITS[:, 1])
    return [{"id": f"{split}_{i:03d}", "split": split,
             "mct_gain": float(row[0]), "pCa0_scale": float(row[1]),
             "serca_scale": float(row[2])} for i, row in enumerate(values)]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = sample(48, 20260929, "train") + sample(16, 20260930, "test")
    with (ROOT / "design.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "domain": "rabbit MCT2024 with MCT, basal ICaL, and SERCA multipliers",
        "design": "independent scrambled 3D Latin hypercube samples for train/test",
        "train_n": 48, "test_n": 16,
        "train_seed": 20260929, "test_seed": 20260930,
        "parameter_ranges": dict(zip(("mct_gain", "pCa0_scale", "serca_scale"), LIMITS.tolist())),
        "loads_CL": [0.02, 0.06, 0.10], "beats": 40, "pacing_ms": 1000,
    }
    (ROOT / "design_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} cell identities to {ROOT / 'design.csv'}")


if __name__ == "__main__":
    main()
