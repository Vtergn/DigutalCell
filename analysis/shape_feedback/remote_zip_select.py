"""Inspect and selectively extract a public ZIP using HTTP byte ranges.

The full sarcomere-image archives are several GB; this avoids downloading
large TIFF movies when only a small, already processed trajectory is needed.
"""

import argparse
import io
import zipfile
from pathlib import Path

import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, chunk_size: int = 8 << 20):
        self.session = requests.Session()
        self.url = url
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=45)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Server did not honor Range; refusing full download")
        self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        self.position = 0
        self.chunk_size = chunk_size
        self.cached_start = -1
        self.cached_bytes = b""
        self.bytes_transferred = len(response.content)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = (offset if whence == io.SEEK_SET else
                         self.position + offset if whence == io.SEEK_CUR else
                         self.size + offset)
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        result = bytearray()
        while len(result) < n:
            if not self.cached_start <= self.position < self.cached_start + len(self.cached_bytes):
                start = (self.position // self.chunk_size) * self.chunk_size
                end = min(start + self.chunk_size, self.size) - 1
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError("Server stopped honoring Range")
                self.cached_start = start
                self.cached_bytes = response.content
                self.bytes_transferred += len(response.content)
            offset = self.position - self.cached_start
            count = min(n - len(result), len(self.cached_bytes) - offset)
            result += self.cached_bytes[offset:offset + count]
            self.position += count
        return bytes(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="17564384")
    parser.add_argument("--archive", default="5kPa.zip")
    parser.add_argument("--extract", default=None, help="Exact archive member path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    url = f"https://zenodo.org/api/records/{args.record}/files/{args.archive}/content"
    reader = RangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        for item in archive.infolist():
            print(f"{item.file_size}\t{item.compress_size}\t{item.filename}")
        if args.extract:
            if not args.output:
                raise ValueError("--output required with --extract")
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(args.extract) as source, target.open("wb") as destination:
                while chunk := source.read(1 << 20):
                    destination.write(chunk)
            print(f"EXTRACTED {target} ({target.stat().st_size} bytes)")
    print(f"HTTP transferred: {reader.bytes_transferred} bytes")


if __name__ == "__main__":
    main()
