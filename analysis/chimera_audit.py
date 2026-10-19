"""Audit cap chimera phenotypes before mechanistic model transfer."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "chimera_experiments" / "capchims_summary.csv"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[row["construct"]].append(row)
    result = []
    for construct, values in sorted(groups.items()):
        record = {
            "construct": construct, "cells": len(values),
            "dates": len({r["date"] for r in values}),
            "date_list": "|".join(sorted({r["date"] for r in values})),
            "voltage_median_mV": float(np.median([float(r["vstep"]) for r in values])),
        }
        for field in ("delay", "wthresh", "Cm", "peakw"):
            array = np.asarray([float(r[field]) for r in values])
            record[f"{field}_median"] = float(np.median(array))
            record[f"{field}_q25"] = float(np.quantile(array, .25))
            record[f"{field}_q75"] = float(np.quantile(array, .75))
        result.append(record)
        print(record)
    with (HERE / "chimera_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
