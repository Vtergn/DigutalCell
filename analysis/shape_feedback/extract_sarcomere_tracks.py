"""Extract only time-resolved sarcomere trajectories from selected Zenodo ZIPs."""
from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data/sarcomere_shape_pilot"
OUT = Path(__file__).resolve().parent


def array(data, key):
    return np.asarray(data[key]["values"], dtype=float)


def main():
    tracks = {}
    summary = {}
    for load in (5, 15, 85):
        tracks[load] = []
        summary[str(load)] = {}
        for path in sorted(SOURCE.glob(f"{load}kPa_cell_*.zip")):
            cell_id = path.stem.rsplit("_", 1)[-1]
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith("_loi_data.json")]
                cells = []
                for n in names:
                    payload = json.load(archive.open(n))
                    t = array(payload, "time")
                    avg = array(payload, "slen_avg")
                    all_sarc = array(payload, "slen")
                    if all_sarc.shape[1] != len(t):
                        raise ValueError(f"unexpected shape for {n}: {all_sarc.shape}")
                    median = float(np.nanmedian(avg))
                    norm = avg / median
                    cells.append((n, t, avg, norm, all_sarc))
                summary[str(load)][cell_id] = {
                    "source_zip": str(path), "loi_count": len(cells),
                    "frame_count": [len(x[1]) for x in cells],
                    "dt_ms": [float(np.nanmedian(np.diff(x[1])) * 1000) for x in cells],
                    "sarcomeres_per_loi": [x[4].shape[0] for x in cells],
                    "normalized_min_by_loi": [float(np.nanmin(x[3])) for x in cells],
                    "normalized_max_by_loi": [float(np.nanmax(x[3])) for x in cells],
                }
                tracks[load].append((cell_id, cells))
    packed = {}
    for load, group in tracks.items():
        for cell_id, cells in group:
            for j, (_, t, avg, norm, all_sarc) in enumerate(cells):
                prefix = f"load{load}_cell{cell_id}_loi{j}"
                packed[f"{prefix}_time_s"] = t
                packed[f"{prefix}_length_um"] = avg
                packed[f"{prefix}_length_normalized"] = norm
                packed[f"{prefix}_sarcomere_length_um"] = all_sarc
    np.savez_compressed(OUT / "sarcomere_tracks.npz", **packed)
    (OUT / "sarcomere_track_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, load in zip(axes, (5, 15, 85)):
        for cell_id, cells in tracks[load]:
            for j, (_, t, avg, norm, all_sarc) in enumerate(cells):
                ax.plot(t, norm, lw=.8, label=f"cell {cell_id}" if j == 0 else None, alpha=.6)
        ax.set_ylabel(f"{load} kPa\nSL / median SL")
        ax.grid(alpha=.18)
        ax.legend(ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(OUT / "sarcomere_tracks.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
