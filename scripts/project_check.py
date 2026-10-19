"""Fast, read-only health check for the active electromechanical project.

This checks provenance and artifact schemas, not biological validity.
Use --deep to re-run the inexpensive target-blind benchmark.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPE = ROOT / "analysis" / "shape_feedback"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main(deep: bool = False, source_only: bool = False) -> None:
    require(ROOT / "README.md")
    require(ROOT / "requirements-core.txt")
    require(ROOT / "docs" / "PROJECT_MAP.md")
    require(ROOT / "docs" / "DATA_DOWNLOADS.md")
    require(ROOT / "analysis" / "torch_native_port" / "prepace_state_v5.mat")
    for name in ("run_embodied_cohort.py", "blind_closed_loop_benchmark.py",
                 "sarcomere_closed_loop.py", "sarcomere_protocol_suite.py"):
        path = require(SHAPE / name)
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    if source_only:
        print("OK: source files, initial state, and documentation are present")
        print("Scope: clone check only; external code and generated cohorts not checked")
        return

    import numpy as np

    require(ROOT / "external" / "SarcomereModel" / "src" / "model")
    require(SHAPE / "sarcomere_closed_loop_viewer" / "index.html")
    require(SHAPE / "sarcomere_closed_loop_viewer" / "viewer_data.js")

    total_cells = 0
    for name in ("embodied_cohort", "embodied_cohort_confirm"):
        folder = require(SHAPE / name)
        manifest = json.loads(require(folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_planned"] == 12 and len(manifest["protocols"]) == 4
        for i in range(1, 13):
            with np.load(require(folder / f"cell_{i:02d}.npz")) as cell:
                assert cell["wave"].shape == (4, 601, 4)
                assert cell["theta"].shape == (4,)
                assert np.isfinite(cell["wave"]).all()
            total_cells += 1

    blind = json.loads(require(SHAPE / "blind_closed_loop" / "results.json").read_text(encoding="utf-8"))
    assert len(blind["per_cell"]) == 24 * 2 * 3
    assert set(blind["summary"]) == {"embodied_cohort", "embodied_cohort_confirm"}
    with np.load(require(SHAPE / "sarcomere_closed_loop.npz")) as one:
        assert one["closed_pulse_state"].shape == (701, 90)
    suite = json.loads(require(SHAPE / "sarcomere_protocol_suite" / "results.json").read_text(encoding="utf-8"))
    assert set(suite["results"]) == {"long_hold", "compound"}
    assert max(suite["single_pulse_regression_max_V_mV_Ca_uM_T_kPa_lambda"]) == 0
    for name in suite["results"]:
        with np.load(require(SHAPE / "sarcomere_protocol_suite" / f"{name}.npz")) as data:
            assert data["state"].shape == (701, 90)
            assert data["local_lambda"].shape == (701, 20)
            assert data["truth"].shape == data["linear"].shape == (701, 4)
    if deep:
        subprocess.run([sys.executable, str(SHAPE / "blind_closed_loop_benchmark.py")],
                       cwd=ROOT, check=True)
    print(f"OK: {total_cells} paired virtual cells; blind benchmark; "
          "90-state single/long/compound protocols; viewer assets")
    print("Scope: artifact integrity and numerical schema, not real-cell validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.deep and args.source_only:
        parser.error("--deep and --source-only cannot be combined")
    main(args.deep, args.source_only)
