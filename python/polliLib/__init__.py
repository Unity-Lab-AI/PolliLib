"""
polliLib: Modular Pollinations client library.

Usage (simple façade):
    from polliLib import (
        PolliClient,
        list_models, get_model_by_name, get_field,
        generate_image, save_image_timestamped, fetch_image,
        generate_text,
        chat_completion, chat_completion_stream, chat_completion_tools,
        transcribe_audio,
        analyze_image_url, analyze_image_file,
        image_feed_stream, text_feed_stream,
    )

Run examples:
    python -m polliLib
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import PolliClient

__all__ = [
    "PolliClient",
    "list_models",
    "get_model_by_name",
    "get_field",
    "generate_image",
    "save_image_timestamped",
    "fetch_image",
    "generate_text",
    "chat_completion",
    "chat_completion_stream",
    "chat_completion_tools",
    "transcribe_audio",
    "analyze_image_url",
    "analyze_image_file",
    "image_feed_stream",
    "text_feed_stream",
    "__version__",
]

__version__ = "1.0.1"


_default_client: Optional[PolliClient] = None


def _client() -> PolliClient:
    global _default_client
    if _default_client is None:
        _default_client = PolliClient()
    return _default_client


def list_models(kind: "ModelType") -> List["Model"]:
    return _client().list_models(kind)


def get_model_by_name(name: str, kind: Optional["ModelType"] = None) -> Optional["Model"]:
    return _client().get_model_by_name(name, kind=kind)


def get_field(model: "Model", field: str, default: Any = None) -> Any:
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


def fetch_image(
    image_url: str,
    *,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 120.0,
    out_path: Optional[str] = None,
    chunk_size: int = 1024 * 64,
) -> bytes | str:
    return _client().fetch_image(
        image_url,
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
):
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
):
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
):
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


def chat_completion_tools(
    messages: List[Dict[str, Any]],
    *,
    tools: List[Dict[str, Any]],
    functions: Optional[Dict[str, "Callable[..., Any]"]] = None,
    tool_choice: Any = "auto",
    model: str = "openai",
    seed: Optional[int] = None,
    private: Optional[bool] = None,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    as_json: bool = False,
    timeout: Optional[float] = 60.0,
    max_rounds: int = 1,
):
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
):
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
):
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
):
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


def image_feed_stream(
    *,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    reconnect: bool = False,
    retry_delay: float = 10.0,
    yield_raw_events: bool = False,
    include_bytes: bool = False,
    include_data_url: bool = False,
):
    return _client().image_feed_stream(
        referrer=referrer,
        token=token,
        timeout=timeout,
        reconnect=reconnect,
        retry_delay=retry_delay,
        yield_raw_events=yield_raw_events,
        include_bytes=include_bytes,
        include_data_url=include_data_url,
    )


def text_feed_stream(
    *,
    referrer: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    reconnect: bool = False,
    retry_delay: float = 10.0,
    yield_raw_events: bool = False,
):
    return _client().text_feed_stream(
        referrer=referrer,
        token=token,
        timeout=timeout,
        reconnect=reconnect,
        retry_delay=retry_delay,
        yield_raw_events=yield_raw_events,
    )
