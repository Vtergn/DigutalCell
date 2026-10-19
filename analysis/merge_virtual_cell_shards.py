"""Combine deterministic independent simulator shards into one cell-indexed HDF5.

Usage: python analysis/merge_virtual_cell_shards.py OUTPUT.h5 SHARD_DIR ...
Only finished shards with cohort.mat and manifest.json are accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    output = Path(sys.argv[1])
    folders = [Path(x) for x in sys.argv[2:]]
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ["V", "Cai_uM", "Ta_kPa", "Tp_kPa", "theta", "success", "split_code"]
    }
    configs = []
    reference = None
    seen_seeds = set()
    for folder in folders:
        config = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if config["seed"] in seen_seeds:
            raise ValueError(f"Repeated sampling seed in {folder}")
        seen_seeds.add(config["seed"])
        if reference is None:
            reference = config
        else:
            for key in ["lambda", "theta_names", "lower_bounds", "upper_bounds", "nBeats"]:
                if config[key] != reference[key]:
                    raise ValueError(f"Shard configuration mismatch: {folder}, {key}")
        with h5py.File(folder / "cohort.mat") as f:
            block = {
                "V": f["V"][()].T,
                "Cai_uM": f["Cai_uM"][()].T,
                "Ta_kPa": f["Ta_kPa"][()].transpose(2, 1, 0),
                "Tp_kPa": f["Tp_kPa"][()].transpose(2, 1, 0),
                "theta": f["theta"][()].T,
                "success": f["success"][()].ravel().astype(bool),
                "split_code": f["split_code"][()].ravel(),
            }
        keep = block["success"]
        if not np.isfinite(block["Ta_kPa"][keep]).all():
            raise ValueError(f"Non-finite tension in shard {folder}")
        for key in arrays:
            arrays[key].append(block[key][keep])
        configs.append({"folder": str(folder.resolve()), "seed": config["seed"],
                        "sampled": len(keep), "passed": int(keep.sum()),
                        "failed_local_cell_ids": (np.flatnonzero(~keep) + 1).tolist()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        for key, blocks in arrays.items():
            data = np.concatenate(blocks, axis=0)
            f.create_dataset(key, data=data, compression="gzip", compression_opts=2, chunks=True)
        f.create_dataset("lambda", data=np.asarray(reference["lambda"]))
        f.create_dataset("time_ms", data=np.arange(arrays["V"][0].shape[1]))
        f.create_dataset("lo", data=np.asarray(reference["lower_bounds"]))
        f.create_dataset("hi", data=np.asarray(reference["upper_bounds"]))
        f.attrs["theta_names_json"] = json.dumps(reference["theta_names"])
        f.attrs["shards_json"] = json.dumps(configs)
        f.attrs["coupling"] = reference["coupling"]
    print(f"Merged {len(folders)} shards, {sum(x['passed'] for x in configs)}/"
          f"{sum(x['sampled'] for x in configs)} passed cells -> {output}")


if __name__ == "__main__":
    main()
