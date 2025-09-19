# polliLib (JavaScript)

A modular Pollinations client library written in modern JavaScript (ESM). Mirrors the Python version’s API and behavior with safe defaults and no server runtime.

- Single import surface with simple façade helpers
- Random 5–8 digit seed by default
- Image, text, chat (incl. streaming + tools), vision, speech‑to‑text, and public feeds
- Optional `referrer` and `token` for endpoints that support them
- Node‑first implementation; can be bundled for static use. FS‑based helpers (saving images, reading local audio/images) are Node‑only.

## Requirements

- Node.js >= 18 (uses global `fetch`, ESM, and `node:test`)

## Install / Use

This repo isn’t published to npm yet. Use the `javascript` folder directly (or copy `polliLib` into your project).

Example (ESM):

```
import {
  PolliClient,
  generate_text, generate_image, save_image_timestamped,
  chat_completion, chat_completion_stream, chat_completion_tools,
  analyze_image_url, analyze_image_file,
  transcribe_audio,
  image_feed_stream, text_feed_stream,
} from './javascript/polliLib/index.js';

// Text
console.log(await generate_text('Explain relativity simply'));

// Image (saves to ./images/<timestamp>.jpeg)
const path = await save_image_timestamped('A beautiful sunset over the ocean');
console.log('Saved:', path);

// Chat
const msgs = [
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'When did the French Revolution start?' },
];
console.log(await chat_completion(msgs));
```

Browser / bundlers:
- The core uses `fetch`. For browser‑only usage, avoid Node‑specific functions (like saving to disk or reading files).
- Use your bundler to include only what you need (tree‑shake mixins).

## API Highlights

- Images: `generate_image`, `save_image_timestamped`, `fetch_image`
- Text: `generate_text`
- Chat: `chat_completion`, `chat_completion_stream`, `chat_completion_tools`
- Vision: `analyze_image_url`, `analyze_image_file`
- STT: `transcribe_audio`
- Feeds: `image_feed_stream`, `text_feed_stream`

All accept `referrer` and/or `token` where supported. Seeds default to a random 5–8 digit integer unless provided.

## Run Tests

From the repo root:

```
node --test javascript/tests
```

Or, with the package script:

```
npm run test
```

Tests are offline: they stub `fetch` and simulate SSE/HTTP responses.

## Notes

- `save_image_timestamped` and `transcribe_audio` use Node’s `fs`; they are not browser APIs.
- `image_feed_stream` can optionally attach raw bytes (`includeBytes: true`) or a base64 data URL (`includeDataUrl: true`) to each feed event.
- If you want a published npm package or CDN build, we can add a simple bundling config (e.g., Rollup) later.
