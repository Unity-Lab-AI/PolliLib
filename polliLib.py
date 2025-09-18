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
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict
import requests

__all__ = [
    "PolliClient",
    "list_models",
    "get_model_by_name",
    "get_field",
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
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.text_url = text_url
        self.image_url = image_url
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

    # ---------- Easy to extend later ----------
    # def generate_text(self, model_name: str, prompt: str, **params) -> str:
    #     raise NotImplementedError
    #
    # def generate_image(self, model_name: str, prompt: str, **params) -> bytes:
    #     raise NotImplementedError

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
    return _client().get(model, field, default)


# -------- Optional: quick demo when executed directly --------
if __name__ == "__main__":
    c = PolliClient()
    print("Text models:", len(c.list_models("text")))
    print("Image models:", len(c.list_models("image")))
    for q in ("flux", "openai", "gemini", "nonexistent"):
        hit = c.get_model_by_name(q)
        print(f"{q!r} ->", hit["name"] if hit else None)
