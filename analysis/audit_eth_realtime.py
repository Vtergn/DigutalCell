"""Read-only audit of extracted ETH real-time FLIM line scans.

The TIFF channel identities are not encoded in the exported XML. Therefore this
script reports channel numbers and acquisition geometry, not calibrated tension.
"""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data/RealtimeFlipperTrImaging/RealtimeFlipperTrImaging/ExtractedLifetimes"
OUT = ROOT / "analysis"


def read_tiff(path):
    with Image.open(path) as image:
        return np.asarray(image)


def scan_settings(xml_path):
    root = ET.parse(xml_path).getroot()
    settings = next(root.iter("ATLConfocalSettingDefinition"))
    spatial = next(
        element for element in root.iter("DimensionDescription")
        if element.attrib.get("Unit") in {"µm", "um"}
    )
    return {
        "scan_mode": settings.attrib["ScanMode"],
        "line_time_s": float(settings.attrib["LineTime"]),
        "nominal_lines": int(settings.attrib["Lines"]),
        "spatial_pixels": int(spatial.attrib["NumberOfElements"]),
        "spatial_extent_um": float(spatial.attrib["Length"]),
    }


def main():
    records = []
    for path in sorted(DATA.glob("*_RT_t0000_ch2.tif")):
        name = path.name.split("_RT_")[0]
        group = "Bleb" if name.startswith("Bleb") else "Cell"
        meta = DATA / "MetaData" / f"{name}_RT_properties.xml"
        if not meta.exists() and name == "Bleb08":
            meta = DATA / "MetaData/BLeb08_RT_properties.xml"
        if not meta.exists():
            # Cell01/02 have extracted TIFFs but no matching XML export.
            meta = DATA / "MetaData/Cell03_RT_properties.xml"
        settings = scan_settings(meta)
        intensity = read_tiff(path)
        # This threshold only identifies lines containing signal. It is not a
        # lifetime or membrane segmentation rule.
        signal_pixels = np.count_nonzero(intensity >= 20, axis=1)
        active = np.flatnonzero(signal_pixels >= 3)
        record = {
            "name": name,
            "group": group,
            **settings,
            "metadata_source": meta.name,
            "tiff_shape": list(intensity.shape),
            "active_lines": int(active.size),
            "active_duration_s": round(float(active.size * settings["line_time_s"]), 2),
            "last_active_s": round(float(active.max() * settings["line_time_s"]), 2),
            "median_signal_pixels_per_active_line": int(np.median(signal_pixels[active])),
            "ch3_median_active_units": float(
                np.median(read_tiff(DATA / f"{name}_RT_t0000_ch3.tif")[intensity >= 20])
            ),
        }
        records.append(record)

    (OUT / "eth_realtime_audit.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, name in enumerate(["Cell03", "Bleb04"]):
        record = next(item for item in records if item["name"] == name)
        active_lines = record["active_lines"]
        intensity = read_tiff(DATA / f"{name}_RT_t0000_ch2.tif")[:active_lines]
        channel3 = read_tiff(DATA / f"{name}_RT_t0000_ch3.tif")[:active_lines]
        duration = record["active_duration_s"]
        extent = (0, duration, record["spatial_extent_um"], 0)
        axes[row, 0].imshow(np.log1p(intensity).T, aspect="auto", extent=extent,
                            cmap="gray", vmin=0, vmax=6)
        axes[row, 0].set(title=f"{name}: log(1 + channel 2)",
                         xlabel="time (s)", ylabel="scan position (µm)")
        image = axes[row, 1].imshow(
            np.ma.masked_where(intensity < 20, channel3 / 1000).T,
            aspect="auto", extent=extent, cmap="viridis", vmin=3.5, vmax=5.5
        )
        axes[row, 1].set(title="channel 3 / 1000, displayed where channel 2 ≥ 20",
                         xlabel="time (s)", ylabel="scan position (µm)")
    fig.colorbar(image, ax=axes[:, 1], label="channel 3 / 1000 (unverified units)")
    fig.savefig(OUT / "eth_realtime_qc.png", dpi=150)
    plt.close(fig)
    print(f"Audited {len(records)} line scans: "
          f"{sum(x['group'] == 'Cell' for x in records)} cells, "
          f"{sum(x['group'] == 'Bleb' for x in records)} blebs")
    for record in records:
        print(record["name"], record["active_duration_s"],
              record["median_signal_pixels_per_active_line"])


if __name__ == "__main__":
    main()
