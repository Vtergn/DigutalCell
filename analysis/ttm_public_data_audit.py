"""Audit the public TTM analysis table for baseline/drug pairing and unit of analysis.

Source: https://gitlab.com/bjvanmeer/ttm-algorithm/-/blob/master/ttm
The GitLab table is a whitespace-delimited R export with an unlabeled row index.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "ttm_public" / "ttm_original"
FIELDS = ("row_index", "area_id", "phase", "dose", "feature", "value", "drug")


def main():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        next(handle)
        rows = []
        for source_line in csv.reader(handle, delimiter=" "):
            if len(source_line) != len(FIELDS):
                raise ValueError(f"Unexpected row width: {source_line[:3]}")
            rows.append(dict(zip(FIELDS, source_line)))
    features = sorted({r["feature"] for r in rows})
    if len(features) != 10:
        raise ValueError(f"Expected 10 features, found {features}")
    records = defaultdict(dict)
    meta = {}
    for r in rows:
        key = (r["area_id"], r["phase"])
        if r["feature"] in records[key]:
            raise ValueError(f"Duplicate feature in {key}")
        records[key][r["feature"]] = r["value"]
        meta[r["area_id"]] = (r["dose"], r["drug"])
    if any(set(record) != set(features) for record in records.values()):
        raise ValueError("Feature set varies between measurement records")
    area_ids = sorted(meta)
    phases = {area: {phase for record_area, phase in records if record_area == area}
              for area in area_ids}
    paired = [area for area in area_ids if phases[area] == {"baseline", "added"}]
    complete = [area for area in paired if all(records[(area, phase)][feature] != "NA"
                                                for phase in ("baseline", "added")
                                                for feature in features)]
    # Original protocol says three areas per well. The ID format has a third
    # field 1/2/3; this grouping is therefore consistent with well identity,
    # but the repository does not explicitly define the ID schema.
    putative_well = lambda area: ":".join([area.split(":")[0], area.split(":")[1],
                                           area.split(":")[3]])
    wells = defaultdict(set)
    for area in area_ids:
        parts = area.split(":")
        if len(parts) != 4:
            raise ValueError(f"Unexpected area ID: {area}")
        wells[putative_well(area)].add(parts[2])
    counts = Counter(map(len, wells.values()))
    per_drug = {}
    for drug in sorted({r["drug"] for r in rows}):
        drug_ids = [area for area in area_ids if meta[area][1] == drug]
        per_drug[drug] = {
            "areas": len(drug_ids),
            "paired_areas": sum(area in paired for area in drug_ids),
            "fully_observed_paired_areas": sum(area in complete for area in drug_ids),
            "putative_wells": len({putative_well(area) for area in drug_ids}),
        }
    result = {
        "source": "public GitLab ttm file, downloaded 2026-09-28; Dryad ZIP listed as 389,910 bytes but its download API returned 401 in this environment",
        "rows": len(rows),
        "features": features,
        "measurement_records": len(records),
        "area_ids": len(area_ids),
        "baseline_records": sum(phase == "baseline" for _, phase in records),
        "added_records": sum(phase == "added" for _, phase in records),
        "baseline_and_added_paired_areas": len(paired),
        "fully_observed_paired_areas": len(complete),
        "fully_observed_paired_putative_wells": len({putative_well(area) for area in complete}),
        "missing_numeric_feature_values": sum(r["value"] == "NA" for r in rows),
        "putative_wells": len(wells),
        "areas_per_putative_well": dict(sorted(counts.items())),
        "drugs": per_drug,
        "unit_warning": "TTM paper reports monolayers, five wells per condition and three imaged areas per well. This table supports area-level pairing and likely well grouping, not single-cell identity or raw AP/Ca/contraction traces.",
    }
    (HERE / "ttm_public_data_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
