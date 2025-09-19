# polliLib.py
"""
A tiny Pollinations helper library.

Goals:
- Simple to use (few, stable entrypoints)
- Safe networking (timeouts)
- Normalized model shapes (dicts), regardless of endpoint quirks
- Easy to extend (future text/image generation, tooling)

Usage:
    from polli import PolliClient

    client = PolliClient()
    text_models = client.list_models("text")      # List[dict]
    image_models = client.list_models("image")    # List[dict]
    m = client.get_model_by_name("flux")          # dict | None (searches both)
    vision = client.get(m, "vision", False)       # safe accessor

    # or use the simple façade:
    from polli import list_models, get_model_by_name
    print(get_model_by_name("openai"))
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict, Iterator, Callable
import requests

__all__ = [
    "PolliClient",
    "list_models",
    "get_model_by_name",
    "get_field",
    "generate_image",
    "save_image_timestamped",
    "generate_text",
    "chat_completion",
    "chat_completion_tools",
    "chat_completion_stream",
    "transcribe_audio",
    "analyze_image_url",
    "analyze_image_file",
    "image_feed_stream",
    "text_feed_stream",
    "__version__",
]

__version__ = "0.1.0"

ModelType = Literal["text", "image"]


class Model(TypedDict, total=False):
    name: str
    description: str
    maxInputChars: int
    reasoning: bool
    community: bool
    tier: str
    aliases: List[str]
    input_modalities: List[str]
    output_modalities: List[str]
    tools: bool
    vision: bool
    audio: bool
    voices: List[str]
    supportsSystemMessages: bool


class PolliClient:
    """
    Minimal client with a small, stable surface:
      - list_models(kind): fetch models for "text" or "image"
      - get_model_by_name(name, kind=None): lookup by name/alias
      - get(model, field, default): safe dict accessor

    The shape of each model dict is normalized so you can safely call .get().
    """

    def __init__(
        self,
        text_url: str = "https://text.pollinations.ai/models",
        image_url: str = "https://image.pollinations.ai/models",
        image_prompt_base: str = "https://image.pollinations.ai/prompt",
        text_prompt_base: str = "https://text.pollinations.ai",
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.text_url = text_url
        self.image_url = image_url
        self.image_prompt_base = image_prompt_base
        self.text_prompt_base = text_prompt_base
        self.timeout = timeout
        self.session = session or requests.Session()

    # ---------- Public API ----------

    @lru_cache(maxsize=4)
    def list_models(self, kind: ModelType) -> List[Model]:
        """Return a cached, normalized list of models for the given kind."""
        url = self._url(kind)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return self._normalize_models(resp.json())

    def get_model_by_name(
        self,
        name: str,
        kind: Optional[ModelType] = None,
        include_aliases: bool = True,
        case_insensitive: bool = True,
    ) -> Optional[Model]:
        """Find a model by name (or alias). Searches both kinds by default."""
        needle = name.casefold() if case_insensitive else name
        kinds: Iterable[ModelType] = (kind,) if kind else ("text", "image")
        for m in self._iter_models(*kinds):
            names = [m.get("name", "")]
            if include_aliases:
                names.extend(m.get("aliases", []) or [])
            if case_insensitive:
                names = [n.casefold() for n in names]
            if needle in names:
                return m
        return None

    @staticmethod
    def get(model: Model, field: str, default: Any = None) -> Any:
        """Safe field accessor with default."""
        return model.get(field, default)

    def refresh_cache(self) -> None:
        """Drop the internal cache so subsequent calls refetch."""
        self.list_models.cache_clear()  # type: ignore[attr-defined]

    # ---------- Internals ----------

    def _url(self, kind: ModelType) -> str:
        return self.text_url if kind == "text" else self.image_url

    def _iter_models(self, *kinds: ModelType) -> Iterable[Model]:
        for k in kinds or ("text", "image"):
            yield from self.list_models(k)

    @staticmethod
    def _normalize_models(raw: Any) -> List[Model]:
        """
        Coerce various API shapes into List[Model] (dicts).
        Accepts:
          - ["flux", "kontext", "turbo"]                 (image endpoint)
          - [{"name": "...", ...}, ...]                  (text endpoint)
          - {"models": [...]}                            (defensive)
        """
        # Unwrap {"models": [...]}
        if isinstance(raw, dict) and "models" in raw and isinstance(raw["models"], list):
            raw = raw["models"]

        if not isinstance(raw, list):
            return []

        out: List[Model] = []
        for item in raw:
            if isinstance(item, str):
                out.append(
                    {
                        "name": item,
                        "aliases": [],
                        "input_modalities": [],
                        "output_modalities": [],
                        "tools": False,
                        "vision": False,
                        "audio": False,
                        "community": False,
                        "supportsSystemMessages": True,
                    }
                )
            elif isinstance(item, dict):
                m: Dict[str, Any] = dict(item)  # shallow copy
                # Normalize common misspellings if they show up
                if "teir" in m and "tier" not in m:
                    m["tier"] = m.pop("teir")
                # Reasonable defaults
                m.setdefault("aliases", [])
                m.setdefault("input_modalities", [])
                m.setdefault("output_modalities", [])
                m.setdefault("tools", False)
                m.setdefault("vision", False)
                m.setdefault("audio", False)
                m.setdefault("community", False)
                m.setdefault("supportsSystemMessages", True)
                out.append(m)  # type: ignore[arg-type]
        return out

    # ---------- Image generation ----------

    def _random_seed(self) -> int:
        """Return a random integer with 5–8 digits (inclusive)."""
        import random

        # Choose digit length between 5 and 8, then sample in that range
        n_digits = random.randint(5, 8)
        low = 10 ** (n_digits - 1)
        high = (10 ** n_digits) - 1
        return random.randint(low, high)

    def _image_prompt_url(self, prompt: str) -> str:
        from urllib.parse import quote

        return f"{self.image_prompt_base}/{quote(prompt)}"

    def _text_prompt_url(self, prompt: str) -> str:
        from urllib.parse import quote

        return f"{self.text_prompt_base}/{quote(prompt)}"

    def generate_image(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        model: str = "flux",
        seed: Optional[int] = None,
        nologo: bool = True,
        image: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        out_path: Optional[str] = None,
        chunk_size: int = 1024 * 64,
    ) -> bytes | str:
        """
        Generate an image from a text prompt via Pollinations and return bytes
        (or write to `out_path` and return the path).

        Defaults:
          - model="flux"
          - width=height=512
          - nologo=True
          - random seed with 5–8 digits when not provided
          - no image (image-to-image) by default
          - no referrer by default

        Args:
            prompt: The text prompt.
            width, height: Output dimensions in pixels.
            model: Image model name (e.g., "flux", "flux-pro", "kontext").
            seed: Optional deterministic seed. If None, a 5–8 digit random seed is used.
            nologo: Whether to suppress logos if supported by backend.
            image: Optional image URL for image-to-image (e.g., for "kontext").
            referrer: Optional referrer string.
            timeout: Request timeout seconds (image gen can take longer; default 300s).
            out_path: If provided, stream to this file and return the path.
            chunk_size: Streaming chunk size when writing to file.

        Returns:
            bytes when out_path is None; otherwise the written out_path string.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        # Basic validation/sanitization
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        # Seed handling
        if seed is None:
            seed = self._random_seed()

        # Build params (minimize overhead and keep consistent types)
        params: Dict[str, Any] = {
            "width": width,
            "height": height,
            "seed": seed,
            "model": model,
            # make sure 'true' is passed as string to match backend expectations
            "nologo": "true" if nologo else "false",
        }
        if image:
            params["image"] = image
        if referrer:
            params["referrer"] = referrer
        if token:
            params["token"] = token

        url = self._image_prompt_url(prompt)
        # Use longer timeout for image generation; fallback to client timeout otherwise
        eff_timeout = timeout if timeout is not None else max(self.timeout, 60.0)

        # Stream when writing to a file to reduce memory usage
        if out_path:
            with self.session.get(url, params=params, timeout=eff_timeout, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            return out_path

        # Otherwise, return the bytes directly
        resp = self.session.get(url, params=params, timeout=eff_timeout)
        resp.raise_for_status()
        return resp.content

    def save_image_timestamped(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        model: str = "flux",
        nologo: bool = True,
        image: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        images_dir: Optional[str] = None,
        filename_prefix: str = "",
        filename_suffix: str = "",
        ext: str = "jpeg",
    ) -> str:
        """
        Convenience helper that generates an image and saves it to
        an images directory using a timestamped filename.

        Path pattern (default): <cwd>/images/<UTC_YYYYMMDD_HHMMSS>.jpeg

        Notes:
          - Seed is always random (5–8 digits); not overridable here by design.
          - Directory is created if missing.
        """
        import os
        import datetime as dt

        # Resolve images directory
        if images_dir is None:
            images_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(images_dir, exist_ok=True)

        # Compose timestamp filename
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_ext = (ext or "jpeg").lstrip(".")
        fname = f"{filename_prefix}{ts}{filename_suffix}.{safe_ext}"
        out_path = os.path.join(images_dir, fname)

        return self.generate_image(
            prompt,
            width=width,
            height=height,
            model=model,
            seed=None,  # always random
            nologo=nologo,
            image=image,
            referrer=referrer,
            token=token,
            timeout=timeout,
            out_path=out_path,
        )

    # ---------- Text generation ----------

    def generate_text(
        self,
        prompt: str,
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        system: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
    ) -> Any:
        """
        Generate text via Pollinations text endpoint and return either a string
        or a parsed JSON object (when as_json=True).

        Defaults:
          - model="openai"
          - random 5–8 digit seed when not provided
          - no referrer by default
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if seed is None:
            seed = self._random_seed()

        params: Dict[str, Any] = {
            "model": model,
            "seed": seed,
        }
        if as_json:
            params["json"] = "true"
        if system:
            params["system"] = system
        if referrer:
            params["referrer"] = referrer
        if token:
            params["token"] = token

        url = self._text_prompt_url(prompt)
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)
        resp = self.session.get(url, params=params, timeout=eff_timeout)
        resp.raise_for_status()

        if as_json:
            # API may return JSON as a string; parse defensively
            import json as _json

            txt = resp.text
            try:
                return _json.loads(txt)
            except Exception:
                # Return raw when not valid JSON to aid debugging
                return txt
        else:
            return resp.text

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
    ) -> Any:
        """
        Chat completion via Pollinations chat endpoint.

        POSTs to {text_prompt_base}/{model} with OpenAI-style messages.

        Defaults:
          - model="openai"
          - random 5–8 digit seed when not provided
          - no referrer by default
          - returns assistant content string by default; full JSON when as_json=True
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")

        if seed is None:
            seed = self._random_seed()

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed,
        }
        if private is not None:
            payload["private"] = bool(private)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token

        url = f"{self.text_prompt_base}/{model}"
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=eff_timeout)
        resp.raise_for_status()
        data = resp.json()

        if as_json:
            return data

        # Extract assistant message content (OpenAI-like structure)
        try:
            return (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
            )
        except Exception:
            pass
        # Fallback to raw text body if structure is unexpected
        return resp.text

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        yield_raw_events: bool = False,
    ) -> Iterator[str]:
        """
        Stream chat completion deltas as text via Server-Sent Events.

        Yields text chunks (delta content). If yield_raw_events=True, yields raw
        event payload lines (JSON strings) instead of extracted content.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")

        if seed is None:
            seed = self._random_seed()

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "stream": True,
        }
        if private is not None:
            payload["private"] = bool(private)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token

        url = f"{self.text_prompt_base}/{model}"
        eff_timeout = timeout if timeout is not None else max(self.timeout, 60.0)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        with self.session.post(url, headers=headers, json=payload, timeout=eff_timeout, stream=True) as resp:
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
                    obj = _json.loads(data)
                    content = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if content:
                        yield content
                except Exception:
                    continue

    def chat_completion_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: List[Dict[str, Any]],
        functions: Optional[Dict[str, Callable[..., Any]]] = None,
        tool_choice: Any = "auto",
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
        max_rounds: int = 1,
    ) -> Any:
        """
        Chat completion with function calling (tools). Executes tool calls via
        `functions` mapping, appends their outputs, and returns the final reply.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of messages")
        if not isinstance(tools, list) or not tools:
            raise ValueError("tools must be a non-empty list of tool specs")

        if seed is None:
            seed = self._random_seed()

        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)

        history: List[Dict[str, Any]] = list(messages)
        rounds = 0
        while True:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": history,
                "seed": seed,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            if private is not None:
                payload["private"] = bool(private)
            if referrer:
                payload["referrer"] = referrer
            if token:
                payload["token"] = token

            resp = self.session.post(url, headers=headers, json=payload, timeout=eff_timeout)
            resp.raise_for_status()
            data = resp.json()

            msg = (data.get("choices", [{}])[0]).get("message", {})
            tool_calls = msg.get("tool_calls", []) or []
            if not tool_calls or rounds >= max_rounds:
                if as_json:
                    return data
                return msg.get("content")

            # Append assistant's tool call request
            history.append(msg)

            # Execute tool calls and append results
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name")
                args_text = tc.get("function", {}).get("arguments", "{}")
                try:
                    import json as _json
                    args = _json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                except Exception:
                    args = {}

                if functions and fn_name in functions:
                    try:
                        result = functions[fn_name](**args) if isinstance(args, dict) else functions[fn_name]()
                    except Exception as e:
                        result = {"error": f"function '{fn_name}' raised: {e}"}
                else:
                    result = {"error": f"no handler for function '{fn_name}'"}

                # Ensure content is a string (JSON is typical)
                if not isinstance(result, str):
                    import json as _json
                    content_str = _json.dumps(result)
                else:
                    content_str = result

                history.append(
                    {
                        "tool_call_id": tc.get("id"),
                        "role": "tool",
                        "name": fn_name,
                        "content": content_str,
                    }
                )

            rounds += 1

    # ---------- Speech to Text ----------

    def transcribe_audio(
        self,
        audio_path: str,
        *,
        question: str = "Transcribe this audio",
        model: str = "openai-audio",
        provider: str = "openai",
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 120.0,
    ) -> Optional[str]:
        """Transcribe local audio using chat with input_audio content."""
        import os, base64

        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        if ext not in {"mp3", "wav"}:
            return None

        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": ext}},
                    ],
                }
            ],
        }
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token

        url = f"{self.text_prompt_base}/{provider}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    # ---------- Vision (image analysis) ----------

    def analyze_image_url(
        self,
        image_url: str,
        *,
        question: str = "What's in this image?",
        model: str = "openai",
        max_tokens: Optional[int] = 500,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        as_json: bool = False,
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token

        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if as_json:
            return data
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    def analyze_image_file(
        self,
        image_path: str,
        *,
        question: str = "What's in this image?",
        model: str = "openai",
        max_tokens: Optional[int] = 500,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        as_json: bool = False,
    ) -> Any:
        import os, base64

        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext not in {"jpeg", "jpg", "png", "gif", "webp"}:
            ext = "jpeg"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:image/{ext};base64,{b64}"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token

        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if as_json:
            return data
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    # ---------- Public feeds (SSE) ----------

    def image_feed_stream(
        self,
        *,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        reconnect: bool = False,
        retry_delay: float = 10.0,
        yield_raw_events: bool = False,
    ) -> Iterator[Any]:
        """
        Stream the public image feed via SSE.
        Yields dicts (parsed JSON) or raw JSON strings when yield_raw_events=True.
        Set reconnect=True to auto-reconnect on errors with retry_delay seconds.
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
                        import json as _json
                        yield _json.loads(data)
                    except Exception:
                        # Skip malformed lines
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
        """
        Stream the public text feed via SSE.
        Yields dicts (parsed JSON) or raw JSON strings when yield_raw_events=True.
        """
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


# -------- Small functional façade (nice for scripts) --------

_default_client: Optional[PolliClient] = None


def _client() -> PolliClient:
    global _default_client
    if _default_client is None:
        _default_client = PolliClient()
    return _default_client


def list_models(kind: ModelType) -> List[Model]:
    return _client().list_models(kind)


def get_model_by_name(name: str, kind: Optional[ModelType] = None) -> Optional[Model]:
    return _client().get_model_by_name(name, kind=kind)


def get_field(model: Model, field: str, default: Any = None) -> Any:
    """Functional wrapper for PolliClient.get() for convenience."""
    return PolliClient.get(model, field, default)


def generate_image(
    prompt: str,
    *,
    width: int = 512,
    height: int = 512,
    model: str = "flux",
    seed: Optional[int] = None,
    nologo: bool = True,
    image: Optional[str] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    out_path: Optional[str] = None,
    chunk_size: int = 1024 * 64,
    ) -> bytes | str:
    """Facade for PolliClient.generate_image(). See client method for docs."""
    return _client().generate_image(
        prompt,
        width=width,
        height=height,
        model=model,
        seed=seed,
        nologo=nologo,
        image=image,
        referrer=referrer,
        token=token,
        timeout=timeout,
        out_path=out_path,
        chunk_size=chunk_size,
    )


def generate_text(
    prompt: str,
    *,
    model: str = "openai",
    seed: Optional[int] = None,
    system: Optional[str] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    as_json: bool = False,
    timeout: Optional[float] = 60.0,
    ) -> Any:
    """Facade for PolliClient.generate_text()."""
    return _client().generate_text(
        prompt,
        model=model,
        seed=seed,
        system=system,
        referrer=referrer,
        token=token,
        as_json=as_json,
        timeout=timeout,
    )


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: str = "openai",
    seed: Optional[int] = None,
    private: Optional[bool] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    as_json: bool = False,
    timeout: Optional[float] = 60.0,
    ) -> Any:
    """Facade for PolliClient.chat_completion()."""
    return _client().chat_completion(
        messages,
        model=model,
        seed=seed,
        private=private,
        referrer=referrer,
        token=token,
        as_json=as_json,
        timeout=timeout,
    )


def chat_completion_stream(
    messages: List[Dict[str, str]],
    *,
    model: str = "openai",
    seed: Optional[int] = None,
    private: Optional[bool] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    yield_raw_events: bool = False,
    ) -> Iterator[str]:
    """Facade for PolliClient.chat_completion_stream()."""
    return _client().chat_completion_stream(
        messages,
        model=model,
        seed=seed,
        private=private,
        referrer=referrer,
        token=token,
        timeout=timeout,
        yield_raw_events=yield_raw_events,
    )


def image_feed_stream(
    *,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    reconnect: bool = False,
    retry_delay: float = 10.0,
    yield_raw_events: bool = False,
) -> Iterator[Any]:
    """Facade for PolliClient.image_feed_stream()."""
    return _client().image_feed_stream(
        referrer=referrer,
        token=token,
        timeout=timeout,
        reconnect=reconnect,
        retry_delay=retry_delay,
        yield_raw_events=yield_raw_events,
    )


def text_feed_stream(
    *,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    reconnect: bool = False,
    retry_delay: float = 10.0,
    yield_raw_events: bool = False,
) -> Iterator[Any]:
    """Facade for PolliClient.text_feed_stream()."""
    return _client().text_feed_stream(
        referrer=referrer,
        token=token,
        timeout=timeout,
        reconnect=reconnect,
        retry_delay=retry_delay,
        yield_raw_events=yield_raw_events,
    )


def chat_completion_tools(
    messages: List[Dict[str, Any]],
    *,
    tools: List[Dict[str, Any]],
    functions: Optional[Dict[str, Callable[..., Any]]] = None,
    tool_choice: Any = "auto",
    model: str = "openai",
    seed: Optional[int] = None,
    private: Optional[bool] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    as_json: bool = False,
    timeout: Optional[float] = 60.0,
    max_rounds: int = 1,
) -> Any:
    """Facade for PolliClient.chat_completion_tools()."""
    return _client().chat_completion_tools(
        messages,
        tools=tools,
        functions=functions,
        tool_choice=tool_choice,
        model=model,
        seed=seed,
        private=private,
        referrer=referrer,
        token=token,
        as_json=as_json,
        timeout=timeout,
        max_rounds=max_rounds,
    )


def transcribe_audio(
    audio_path: str,
    *,
    question: str = "Transcribe this audio",
    model: str = "openai-audio",
    provider: str = "openai",
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 120.0,
) -> Optional[str]:
    """Facade for PolliClient.transcribe_audio()."""
    return _client().transcribe_audio(
        audio_path,
        question=question,
        model=model,
        provider=provider,
        referrer=referrer,
        token=token,
        timeout=timeout,
    )


def analyze_image_url(
    image_url: str,
    *,
    question: str = "What's in this image?",
    model: str = "openai",
    max_tokens: Optional[int] = 500,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 60.0,
    as_json: bool = False,
) -> Any:
    """Facade for PolliClient.analyze_image_url()."""
    return _client().analyze_image_url(
        image_url,
        question=question,
        model=model,
        max_tokens=max_tokens,
        referrer=referrer,
        token=token,
        timeout=timeout,
        as_json=as_json,
    )


def analyze_image_file(
    image_path: str,
    *,
    question: str = "What's in this image?",
    model: str = "openai",
    max_tokens: Optional[int] = 500,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 60.0,
    as_json: bool = False,
) -> Any:
    """Facade for PolliClient.analyze_image_file()."""
    return _client().analyze_image_file(
        image_path,
        question=question,
        model=model,
        max_tokens=max_tokens,
        referrer=referrer,
        token=token,
        timeout=timeout,
        as_json=as_json,
    )

def save_image_timestamped(
    prompt: str,
    *,
    width: int = 512,
    height: int = 512,
    model: str = "flux",
    nologo: bool = True,
    image: Optional[str] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    images_dir: Optional[str] = None,
    filename_prefix: str = "",
    filename_suffix: str = "",
    ext: str = "jpeg",
) -> str:
    """Facade for PolliClient.save_image_timestamped()."""
    return _client().save_image_timestamped(
        prompt,
        width=width,
        height=height,
        model=model,
        nologo=nologo,
        image=image,
        referrer=referrer,
        token=token,
        timeout=timeout,
        images_dir=images_dir,
        filename_prefix=filename_prefix,
        filename_suffix=filename_suffix,
        ext=ext,
    )


 
# -------- Examples when executed directly --------
if __name__ == "__main__":
    c = PolliClient()

    # Model listing examples
    print("Text models:", len(c.list_models("text")))
    print("Image models:", len(c.list_models("image")))
    for q in ("flux", "openai", "gemini", "nonexistent"):
        hit = c.get_model_by_name(q)
        print(f"{q!r} ->", hit["name"] if hit else None)

    # Text generation example (random seed by default)
    print("\n--- Text Generation Example ---")
    txt = generate_text("Explain the theory of relativity simply")
    print(txt)

    # Image generation example (saves to ./images/<timestamp>.jpeg, random seed)
    print("\n--- Image Generation Example ---")
    saved = save_image_timestamped("A beautiful sunset over the ocean")
    print("Image saved to:", saved)

    # Chat completion example (OpenAI-style messages; random seed by default)
    print("\n--- Chat Completion Example ---")
    msgs = [
        {"role": "system", "content": "You are a helpful historian."},
        {"role": "user", "content": "When did the French Revolution start?"},
    ]
    reply = chat_completion(msgs)
    print("Assistant:", reply)

    # Chat completion streaming example (prints chunks as they arrive)
    print("\n--- Chat Completion Streaming Example ---")
    stream_msgs = [
        {"role": "user", "content": "Tell me a story that unfolds slowly."}
    ]
    for part in chat_completion_stream(stream_msgs):
        print(part, end="", flush=True)
    print("\n[stream complete]")

    # Function calling example
    print("\n--- Function Calling Example ---")
    def get_current_weather(location: str, unit: str = "celsius"):
        if "tokyo" in (location or "").lower():
            return {"location": location, "temperature": "15", "unit": unit, "description": "Cloudy"}
        return {"location": location, "temperature": "unknown"}

    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    fc_messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    fc_reply = chat_completion_tools(fc_messages, tools=tool_spec, functions={"get_current_weather": get_current_weather})
    print("Assistant:", fc_reply)

    # Vision example (URL)
    print("\n--- Vision (URL) Example ---")
    vision_reply = analyze_image_url(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1024px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        question="Describe the main subject.",
    )
    print(vision_reply)

    # Speech-to-Text example (skips if sample.wav not present)
    try:
        import os
        sample = os.path.join(os.getcwd(), "sample.wav")
        if os.path.exists(sample):
            print("\n--- Speech-to-Text Example ---")
            transcript = transcribe_audio(sample)
            print(transcript)
        else:
            print("\n[Skipping Speech-to-Text; no sample.wav found]")
    except Exception as e:
        print("\n[Speech-to-Text error]", e)

    # Real-time public feeds (endless). Uncomment to run.
    # print("\n--- Public Image Feed (endless) ---")
    # for event in image_feed_stream(reconnect=True):
    #     # event is a dict with fields like: prompt, imageURL, model, seed
    #     print("New image:", event.get("prompt"), event.get("imageURL"))
    #     # Optionally break after some items
    #     # break

    # print("\n--- Public Text Feed (endless) ---")
    # for event in text_feed_stream(reconnect=True):
    #     # event contains model, messages, response, etc.
    #     print("Model:", event.get("model"))
    #     msgs = event.get("messages") or []
    #     user = next((m for m in msgs if m.get("role") == "user"), None)
    #     if user and user.get("content"):
    #         content = user["content"]
    #         preview = content if isinstance(content, str) else str(content)
    #         print("User:", (preview[:80] + ("..." if len(preview) > 80 else "")))
    #     print("Response:", (event.get("response", "")[:100] + "..."))
