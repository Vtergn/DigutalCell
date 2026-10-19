"""Read only LIF XML headers and TIFF tags for FLIM channel units."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data/RealtimeFlipperTrImaging/RealtimeFlipperTrImaging"
OUT = ROOT / "analysis/eth_realtime_channel_metadata_audit.json"


def lif_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header = handle.read(13)
        assert int.from_bytes(header[:4], "little") == 112
        chars = int.from_bytes(header[9:13], "little")
        xml = handle.read(chars*2).decode("utf-16-le", errors="replace")
    tags = re.findall(r"<ChannelDescription\b[^>]*>", xml)
    names = re.findall(r'NameOfMeasuredQuantity="([^"]*)"', " ".join(tags))
    units = re.findall(r'Unit="([^"]*)"', " ".join(tags))
    luts = re.findall(r'LUTName="([^"]*)"', " ".join(tags))
    return {"file": path.name, "xml_characters": chars,
            "channel_descriptions": len(tags),
            "nonempty_measured_quantity_names": sorted(set(x for x in names if x)),
            "nonempty_channel_units": sorted(set(x for x in units if x)),
            "lut_counts": dict(Counter(luts).most_common(8))}


def main() -> None:
    records = [lif_header(path) for path in sorted(DATA.glob("*.lif"))]
    sample = DATA / "ExtractedLifetimes/Cell03_RT_t0000_ch3.tif"
    with Image.open(sample) as image:
        tiff = {"file": sample.name, "description_tag": image.tag_v2.get(270),
                "software_tag": image.tag_v2.get(305), "tag_ids": list(image.tag_v2.keys())}
    output = {"lif_files": records, "tiff_sample": tiff,
              "conclusion": "Header/channel metadata alone does not establish the physical unit or lifetime identity of exported ch3 when name and unit attributes are empty."}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"lif_count": len(records),
                      "files_with_nonempty_channel_unit": sum(bool(x["nonempty_channel_units"]) for x in records),
                      "files_with_nonempty_quantity_name": sum(bool(x["nonempty_measured_quantity_names"]) for x in records),
                      "tiff_description": tiff["description_tag"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
