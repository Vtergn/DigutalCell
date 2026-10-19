"""Inspect the central directory of a large remote Zenodo ZIP via HTTP ranges.

Does not download the archive payload or extract files. Reports filenames,
sizes, extension counts, and likely electrophysiology paths for manual audit.
"""

from __future__ import annotations

import collections
import csv
import io
import json
import zipfile
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
RECORD = "https://zenodo.org/api/records/17644763"


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.session = requests.Session()
        self.url = url
        self.size = size
        self.position = 0
        self.bytes_downloaded = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (offset if whence == 0 else
                         self.position+offset if whence == 1 else self.size+offset)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size-self.position
        if size == 0 or self.position >= self.size:
            return b""
        stop = min(self.position+size, self.size)-1
        response = self.session.get(self.url,
                                    headers={"Range": f"bytes={self.position}-{stop}"},
                                    timeout=90)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Server did not honor HTTP Range; refusing full ZIP download")
        data = response.content
        self.bytes_downloaded += len(data)
        self.position += len(data)
        if len(data) != stop-(self.position-len(data))+1:
            raise IOError("Truncated remote ZIP range")
        return data


def main() -> None:
    record = requests.get(RECORD, timeout=30).json()
    files = record["files"]
    if len(files) != 1 or not files[0]["key"].lower().endswith(".zip"):
        raise RuntimeError("Zenodo file layout changed; inspect metadata before download")
    remote = RemoteZip(files[0]["links"]["self"], int(files[0]["size"]))
    with zipfile.ZipFile(remote) as archive:
        info = archive.infolist()
    entries = []
    extensions = collections.Counter()
    for item in info:
        suffix = Path(item.filename).suffix.lower() if not item.is_dir() else "<directory>"
        extensions[suffix] += 1
        entries.append({"path": item.filename, "bytes": item.file_size,
                        "compressed_bytes": item.compress_size, "extension": suffix})
    with (HERE / "zenodo_17644763_manifest.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    candidates = [entry for entry in entries if any(
        keyword in entry["path"].lower() for keyword in
        ("electro", "patch", "poking", "poke", "pressure", "current", "trace", "fichier"))]
    payload = [entry for entry in entries if not entry["path"].startswith("__MACOSX/")
               and entry["extension"] in {".mat", ".nd2", ".obf"}]
    modalities = {name: {"files": sum(
        name.lower() in entry["path"].lower() for entry in payload),
                         "uncompressed_bytes": sum(int(entry["bytes"])
                                                   for entry in payload if
                                                   name.lower() in entry["path"].lower())}
                  for name in ("MINFLUX", "Confocal Images", "STED imaging files")}
    output = {"source": RECORD, "zip_name": files[0]["key"],
              "zip_bytes": int(files[0]["size"]),
              "range_bytes_downloaded": remote.bytes_downloaded,
              "entries": len(entries), "extensions": extensions,
              "payload_files": len(payload), "modalities": modalities,
              "candidate_paths": candidates[:100],
              "candidate_count": len(candidates)}
    (HERE / "zenodo_17644763_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
