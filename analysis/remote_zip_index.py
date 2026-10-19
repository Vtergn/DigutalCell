"""List a large public ZIP via HTTP ranges without downloading the archive."""
from __future__ import annotations

import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://ndownloader.figshare.com/files/23455241"
SIZE = 3040801158


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.bytes_transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, SIZE + offset][whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = SIZE - self.pos
        if n == 0:
            return b""
        end = min(SIZE, self.pos + n) - 1
        req = urllib.request.Request(URL, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError(f"Expected partial response, got {response.status}")
            data = response.read()
        self.pos += len(data)
        self.bytes_transferred += len(data)
        return data


if __name__ == "__main__":
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        for entry in archive.infolist():
            print(f"{entry.file_size:>12} {entry.compress_size:>12} {entry.filename}")
    print(f"INDEX_BYTES_TRANSFERRED={remote.bytes_transferred}")


def extract_small_code(destination: Path):
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        for entry in archive.infolist():
            if not entry.filename.startswith("ModelCode/") or not (entry.filename.endswith((".m", ".txt")) or entry.filename == "ModelCode/X0ORD_CL1000.mat"):
                continue
            target = destination / Path(entry.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
            print(f"EXTRACTED {target} {entry.file_size}")
    print(f"CODE_BYTES_TRANSFERRED={remote.bytes_transferred}")
