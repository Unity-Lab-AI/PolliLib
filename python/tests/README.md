## Tests for polliLib

Offline pytest suite for the modular Pollinations client.

- Uses stubbed `requests.Session` (FakeSession) and FakeResponse to avoid network.
- Covers seeds, text, chat (incl. streaming + tools), images (gen/fetch), feeds, STT, and vision.

### Install and Run

```
python -m pip install -r ../requirements.txt
pytest .
```

You can also run from the repo root:
```
python -m pip install -r python/requirements.txt
pytest python/tests
```

### Structure

- `conftest.py` – shared fakes and path bootstrap
- `test_text_chat.py` – text, chat, streaming, function tools
- `test_images_feeds.py` – image generation/fetch and public feeds
- `test_stt_vision.py` – speech-to-text and vision

### Notes

- Tests do not require network access.
- Add new tests by following the established FakeSession pattern.
