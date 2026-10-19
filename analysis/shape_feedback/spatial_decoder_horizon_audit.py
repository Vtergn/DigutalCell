"""Apply frozen 10-frame spatial-decoder coefficients at other horizons."""
import json
from pathlib import Path

import numpy as np
import spatial_decoder_gate as decoder

HERE = Path(__file__).resolve().parent


def main():
    base = json.loads((HERE / "spatial_decoder_gate.json").read_text(encoding="utf-8"))
    scores = {}
    for horizon in (5, 10, 20):
        decoder.HORIZON = horizon
        tracks = decoder.collect()
        rows = []
        for load, cells in tracks.items():
            coeff = base["selected_coefficients"][str(load)]["coupled"]
            for cell, lois in cells.items():
                row = {"cell": cell, "load_kpa": load}
                for model, model_coeff in (("uniform", (0, 0, 0)),
                                           ("coupled", coeff)):
                    row[model] = decoder.cell_scores(lois, model, model_coeff)[0]
                rows.append(row)
        scores[str(horizon)] = {
            "cell_scores": rows,
            "cell_macro_uniform_mae": float(np.mean([x["uniform"]["mae"] for x in rows])),
            "cell_macro_coupled_mae": float(np.mean([x["coupled"]["mae"] for x in rows])),
            "cells_improved": int(sum(x["coupled"]["mae"] < x["uniform"]["mae"]
                                      for x in rows)),
        }
    result = {"training_horizon_frames": 10,
              "evaluation_horizon_frames": [5, 10, 20],
              "all_coefficients_frozen_from_10_frame_training": True,
              "same_six_cells_and_whole_stiffness_splits": True,
              "scores": scores}
    (HERE / "spatial_decoder_horizon_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: {j: v[j] for j in (
        "cell_macro_uniform_mae", "cell_macro_coupled_mae", "cells_improved")}
        for k, v in scores.items()}, indent=2))


if __name__ == "__main__":
    main()
