from __future__ import annotations

from typing import Any, Dict, Iterator, Optional


class FeedsMixin:
    def image_feed_stream(
        self,
        *,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        reconnect: bool = False,
        retry_delay: float = 10.0,
        yield_raw_events: bool = False,
        include_bytes: bool = False,
        include_data_url: bool = False,
    ) -> Iterator[Any]:
        """
        Stream the public image feed via SSE.
        - Yields dicts or raw JSON strings when yield_raw_events=True.
        - include_bytes -> add 'image_bytes' to each dict
        - include_data_url -> add 'image_data_url' (base64) to each dict
        """
        feed_url = "https://image.pollinations.ai/feed"

        def _connect() -> Iterator[Any]:
            params: Dict[str, Any] = {}
            if referrer:
                params["referrer"] = referrer
            if token:
                params["token"] = token
            headers = {"Accept": "text/event-stream"}
            with self.session.get(feed_url, params=params, headers=headers, stream=True, timeout=timeout or self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    line = raw.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    if yield_raw_events:
                        yield data
                        continue
                    try:
                        import json as _json, base64 as _b64
                        ev = _json.loads(data)
                        if include_data_url or include_bytes:
                            img_url = ev.get("imageURL") or ev.get("image_url")
                            if img_url:
                                r = self.session.get(img_url, timeout=timeout or self.timeout)
                                r.raise_for_status()
                                content = r.content
                                if include_data_url:
                                    ctype = r.headers.get("Content-Type", "image/jpeg")
                                    b64 = _b64.b64encode(content).decode("utf-8")
                                    ev["image_data_url"] = f"data:{ctype};base64,{b64}"
                                elif include_bytes:
                                    ev["image_bytes"] = content
                        yield ev
                    except Exception:
                        continue

        if not reconnect:
            yield from _connect()
            return

        import time as _time
        while True:
            try:
                for item in _connect():
                    yield item
            except Exception:
                pass
            _time.sleep(retry_delay)

    def text_feed_stream(
        self,
        *,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        reconnect: bool = False,
        retry_delay: float = 10.0,
        yield_raw_events: bool = False,
    ) -> Iterator[Any]:
        feed_url = "https://text.pollinations.ai/feed"

        def _connect() -> Iterator[Any]:
            params: Dict[str, Any] = {}
            if referrer:
                params["referrer"] = referrer
            if token:
                params["token"] = token
            headers = {"Accept": "text/event-stream"}
            with self.session.get(feed_url, params=params, headers=headers, stream=True, timeout=timeout or self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    line = raw.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    if yield_raw_events:
                        yield data
                        continue
                    try:
                        import json as _json
                        yield _json.loads(data)
                    except Exception:
                        continue

        if not reconnect:
            yield from _connect()
            return

        import time as _time
        while True:
            try:
                for item in _connect():
                    yield item
            except Exception:
                pass
            _time.sleep(retry_delay)

