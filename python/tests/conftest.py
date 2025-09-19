import os
import sys
import json

# Ensure the package path is importable: add the parent 'python' directory
CURRENT_DIR = os.path.dirname(__file__)
PYTHON_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)


class FakeResponse:
    def __init__(self, *, status=200, text=None, json_data=None, stream_lines=None, content=b"", headers=None, content_chunks=None):
        self.status_code = status
        self._text = text
        self._json = json_data
        self._lines = stream_lines or []
        self.content = content
        self.headers = headers or {}
        self._chunks = content_chunks
        self._closed = False

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self._text or "{}")

    @property
    def text(self):
        return self._text or ""

    def iter_lines(self, decode_unicode=False):
        for ln in self._lines:
            yield ln

    def iter_content(self, chunk_size=8192):
        if self._chunks is not None:
            for ch in self._chunks:
                yield ch
            return
        if self.content:
            yield self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._closed = True


class FakeSession:
    def __init__(self):
        self.last_get = None
        self.last_post = None

    def get(self, url, **kw):
        self.last_get = (url, kw)
        return FakeResponse(status=200, text="ok")

    def post(self, url, headers=None, json=None, **kw):
        self.last_post = (url, headers or {}, json or {}, kw)
        return FakeResponse(json_data={"choices": [{"message": {"content": "ok"}}]})

