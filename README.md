# PolliLib

A modular, language-agnostic client library for Pollinations AI — with first-class Python and JavaScript implementations, a shared API AST, and tests. PolliLib focuses on clear defaults, portability, and symmetry across languages.

## Highlights
- Unified client design in Python and JavaScript (ESM).
- Image generation, text generation, chat completions (non-stream + SSE stream), tools/function-calling, vision (image URL + local file), and speech-to-text.
- Public feeds (SSE) for images and text with optional image bytes or data URLs.
- Random seed defaults (5–8 digits) applied consistently across image and text/chat APIs.
- Optional `referrer` and `token` supported on all endpoints that accept them.
- Language-agnostic API AST in `/AST` to keep the two implementations in sync.

## Repository Structure
```
AST/                    # Language-agnostic API AST (JSON) + README
javascript/             # JavaScript ESM library + tests
  polliLib/             # Core modules (base, images, text, chat, vision, stt, feeds)
  tests/                # Node test suite (node:test)
  package.json          # Library metadata (no server, library-only)
python/                 # Python library + tests
  polliLib/             # Core modules and __main__ examples
  tests/                # pytest suite
  requirements.txt      # Dev/test dependencies
AGENTS.md               # Repo guidance for agents (stage+commit only)
README.md               # You are here
```

## Quick Start

### Python
- Requirements: Python 3.10+.
- Install deps: `python -m pip install -r python/requirements.txt`
- Import via module path (without packaging):

```python
import os, sys
sys.path.append(os.path.join(os.getcwd(), 'python'))

from polliLib import PolliClient, generate_image, generate_text

# Image (defaults: model=flux, 512x512, nologo=true, random seed)
path = generate_image(
    'A serene lake at sunrise, photorealistic, 35mm',
    out_path='images/example.jpeg',
)
print('Saved image to', path)

# Text (default model=openai, random seed)
text = generate_text('Write a haiku about lakes at sunrise')
print(text)
```

- Tests: `pytest python/tests`
- Examples: `python -m polliLib` (runs minimal examples; streaming feeds are commented to avoid endless loops).

### JavaScript (Node, ESM)
- Requirements: Node 18+ (global `fetch` available). For older Node, pass a `fetch` in the client constructor.
- Import and use:

```js
import {
  PolliClient,
  generate_image,
  generate_text,
  chat_completion_stream,
} from './javascript/polliLib/index.js';

// Image (defaults: model=flux, 512x512, nologo=true, random seed)
const imgPath = await generate_image(
  'A serene lake at sunrise, photorealistic, 35mm',
  { outPath: 'images/example.jpeg' }
);
console.log('Saved image to', imgPath);

// Text (default model=openai, random seed)
const text = await generate_text('Write a haiku about lakes at sunrise');
console.log(text);

// Chat streaming (SSE)
for await (const chunk of chat_completion_stream([
  { role: 'user', content: 'Tell me about 35mm photography aesthetics' }
])) {
  process.stdout.write(chunk);
}
```

- Tests: `node --test javascript/tests`

## API Overview (Shared Semantics)
- Seeds: If omitted, a random 5–8 digit seed is generated.
- Images: Defaults — `model=flux`, `width=512`, `height=512`, `nologo=true`. When `out_path/outPath` is provided, bytes stream to disk; otherwise the function returns binary data (Python `bytes`, JS `Buffer`).
- Text: `model='openai'` by default. `as_json/asJson` optionally parses JSON responses with a string fallback.
- Chat: Non-streaming returns assistant content (or full JSON when requested). Streaming yields content chunks via SSE; terminator is `[DONE]`.
- Tools/Function-calling: Provide tool specs and optional local functions; tool calls are executed locally and appended to conversation history up to `max_rounds`.
- Vision: Accepts image URLs or local files (encoded as data URLs) and returns content; optional JSON output.
- Speech-to-Text: Accepts `mp3` or `wav`, encodes to base64 input_audio, returns transcribed text.
- Feeds: Public image/text feeds via SSE. Optional byte/data URL inclusion for image events.
- Optional `referrer` and `token`: Include them where supported; both may be supplied.

## The AST (/AST)
- Purpose: Keep Python and JavaScript APIs aligned, document contracts, and simplify cross-language porting.
- Files: `polli.ast.json` (manifest + shared types) and per-module ASTs (`base`, `images`, `text`, `chat`, `vision`, `stt`, `feeds`, `client`).
- Maintenance rules:
  - Update AST whenever public API surfaces change (names, params/defaults, returns, streaming contracts, errors).
  - Record Python snake_case vs JS camelCase and timeout units (seconds vs milliseconds).
  - Validate behavior with both test suites before committing AST changes.

## Development and Testing
- Python tests: `python -m pip install -r python/requirements.txt && pytest python/tests`
- JS tests: `node --test javascript/tests`
- Formatting: Follow existing style (no new tooling unless already configured).
- Do not push from automation. Follow AGENTS.md: stage and commit only; maintainers will push.

## Contribution Guidelines
- See `CONTRIBUTING.md` for detailed guidance.
- Highlights:
  - Stage + commit only (no pushes). Optionally create tags if requested.
  - Update `/AST` whenever changing public APIs.
  - Add or update tests in `python/tests` and `javascript/tests`.
  - Keep parameters and defaults consistent across languages.
  - Document new features in language READMEs and the root README.

## Project Notes
- Packaging: The repo currently favors direct source usage for Python/JS. If publishing to PyPI/npm is desired, add packaging metadata and update READMEs accordingly.
- Environment: For Python imports without packaging, ensure your `PYTHONPATH` includes the `python` folder (as shown above).
- Node fetch: On Node <18 or in browsers lacking `fetch`, pass a `fetch` implementation to `new PolliClient({ fetch })`.

## License
- See repository terms (no license file included). If you need explicit licensing, open an issue to discuss.

## Acknowledgments
- Built for Pollinations AI. This project aims to provide a clean, modular client with mirrored Python and JavaScript APIs and a shared AST to accelerate cross-language evolution.
