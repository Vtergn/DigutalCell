"""Fetch only one raw cardiomyocyte TIFF from a 5.5 GB Zenodo ZIP.

The original TIFF is temporarily extracted, then an unaltered-intensity,
resized, 1.2-3.0 s animated WebP is saved for the local viewer. No entire
archive download and no retained 145 MB TIFF are required.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from remote_zip_inspect import RangeFile, URL

HERE = Path(__file__).resolve().parent
ENTRY = "5kPa/20210915_5kPa_dish5_day27_lif_Series002.tif"


def main():
    with RangeFile(URL, block_size=2 << 20) as remote:
        with zipfile.ZipFile(remote) as archive:
            info = archive.getinfo(ENTRY)
            print("Selected raw movie", info.filename, info.file_size / 1e6,
                  "MB; remote archive 5.54 GB is not downloaded", flush=True)
            with tempfile.TemporaryFile() as temp, archive.open(info) as source:
                shutil.copyfileobj(source, temp, length=2 << 20)
                temp.seek(0)
                movie = Image.open(temp)
                print("TIFF", movie.size, "frames", movie.n_frames,
                      "mode", movie.mode, flush=True)
                # Registered LOI movie runs at 60.93 Hz in the saved trace.
                start, stop = 73, 183
                frames = []
                for i in range(start, min(stop + 1, movie.n_frames)):
                    movie.seek(i)
                    raw = np.asarray(movie).astype(np.float32)
                    if raw.ndim == 3:
                        raw = raw[..., 0]
                    # One fixed display scale across all frames; avoids
                    # per-frame contrast changes that mimic contraction.
                    if i == start:
                        display_hi = float(np.percentile(raw, 99.7))
                        display_hi = max(display_hi, 1.0)
                    display = np.uint8(np.clip(raw / display_hi * 255, 0, 255))
                    frame = Image.fromarray(display, mode="L")
                    if frame.width > 920:
                        frame.thumbnail((920, 640), Image.Resampling.LANCZOS)
                    frames.append(frame.convert("RGB"))
                output = HERE / "real_5kpa_cell01_1p2-3p0s.webp"
                frames[0].save(output, save_all=True, append_images=frames[1:],
                               duration=16, loop=0, format="WEBP", quality=78,
                               method=4)
                manifest = {
                    "source_url": URL, "zip_entry": ENTRY,
                    "raw_tiff_bytes": info.file_size,
                    "frames": len(frames), "source_frame_start": start,
                    "source_frame_end": min(stop, movie.n_frames - 1),
                    "frame_duration_ms_in_viewer": 16,
                    "display_scale": "fixed using first displayed frame 99.7th percentile",
                    "display_high": display_hi,
                    "resized_width_pixels": frames[0].width,
                    "raw_tiff_retained": False,
                }
                (HERE / "real_movie_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8")
                print("Wrote", output, output.stat().st_size / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main()
