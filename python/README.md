# polliLib (Python)

A modular Pollinations helper library with a small, stable surface and safe defaults.

- Simple façade for common tasks (one import)
- Randomized seeds by default (5–8 digits)
- Image, text, chat (incl. streaming + tools), vision, speech-to-text, and public feeds
- Optional `referrer` and `token` support across endpoints

## Install

This repo isn’t packaged yet. The simplest path is to use the `python` folder on `PYTHONPATH` and install minimal deps.

```
python -m pip install -r python/requirements.txt
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

Windows (PowerShell):
```
python -m pip install -r python/requirements.txt
$env:PYTHONPATH = "$(Get-Location)\python" + ';' + $env:PYTHONPATH
```

## Quick Start

```
from polliLib import (
    PolliClient,
    generate_text, generate_image, save_image_timestamped,
    chat_completion, chat_completion_stream, chat_completion_tools,
    analyze_image_url, analyze_image_file,
    transcribe_audio,
    image_feed_stream, text_feed_stream,
)

# Text
print(generate_text("Explain relativity simply"))

# Image (saves to ./images/<timestamp>.jpeg)
saved = save_image_timestamped("A beautiful sunset over the ocean")
print("Saved:", saved)

# Chat (OpenAI-style messages)
msgs = [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "When did the French Revolution start?"},
]
print(chat_completion(msgs))
```

## Run Examples

Run the package as a module to execute integrated examples:

```
python -m polliLib
```

This prints model counts, runs a text example, saves an image, runs a chat completion + streaming, a function-calling example, a vision example, and a speech-to-text example (if `sample.wav` exists). Public feed examples are included but commented out because they’re endless streams.

## API Highlights

- Images: `generate_image`, `save_image_timestamped`, `fetch_image`
- Text: `generate_text`
- Chat: `chat_completion`, `chat_completion_stream`, `chat_completion_tools`
- Vision: `analyze_image_url`, `analyze_image_file`
- STT: `transcribe_audio`
- Feeds: `image_feed_stream`, `text_feed_stream`

All accept `referrer` and/or `token` where supported. Seeds default to a random 5–8 digit integer unless provided.

## Directory Layout

- `polliLib/`
  - `__init__.py` – single import surface & facades
  - `__main__.py` – runnable examples
  - `client.py` – PolliClient (composes mixins)
  - `base.py` – core utilities, model list/lookup, helpers
  - `images.py`, `text.py`, `chat.py`, `vision.py`, `stt.py`, `feeds.py`
- `tests/` – pytest suite (offline via stubbed sessions)

## Testing

```
python -m pip install -r python/requirements.txt
pytest python/tests
```

Tests use `FakeSession` to avoid real network calls. You can extend tests by following existing patterns.

## Notes

- Image feed helpers can optionally attach raw bytes (`include_bytes=True`) or a base64 data URL (`include_data_url=True`) for easy display.
- If you want a packaged install (`pip install -e .`), we can add a `pyproject.toml` later.
