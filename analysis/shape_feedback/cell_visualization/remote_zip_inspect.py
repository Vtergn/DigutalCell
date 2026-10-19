"""Inspect a large Zenodo ZIP by HTTP byte ranges without downloading it."""
from __future__ import annotations

import io
import re
import zipfile

import requests

URL = "https://zenodo.org/api/records/17564384/files/5kPa.zip/content"


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 1 << 20):
        self.url, self.block_size, self.position = url, block_size, 0
        self.session = requests.Session()
        r = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("Server does not support byte ranges")
        self.size = int(re.search(r"/(\d+)$", r.headers["Content-Range"]).group(1))
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(whence)
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        parts = []
        while n:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                end = min(start + self.block_size, self.size) - 1
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"Range request failed: {r.status_code}")
                self.cache[block] = r.content
                if len(self.cache) > 4:
                    self.cache.pop(next(iter(self.cache)))
            local = self.position - block * self.block_size
            chunk = self.cache[block][local:local+n]
            if not chunk:
                raise EOFError("Short range response")
            parts.append(chunk)
            self.position += len(chunk)
            n -= len(chunk)
        return b"".join(parts)


if __name__ == "__main__":
    with RangeFile(URL) as remote:
        with zipfile.ZipFile(remote) as archive:
            entries = archive.infolist()
            print("entries", len(entries))
            for item in entries[:80]:
                print(item.filename, item.file_size, item.compress_size)
