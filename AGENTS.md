# PolliLib Agent Guide

This document explains how agents (and contributors) should work in this repository. It covers workflow expectations, repo structure, code conventions, testing, and how to extend functionality safely. Follow this guide whenever you make changes here.

## Prime Directive

- Stage and commit your changes. Do not push. The maintainer handles pushes and remotes. If you want to mark a checkpoint, you may optionally create a tag.
- You are expected to act as the full lead developer when working in this codebase: make cohesive, well‑documented changes, keep the surface area stable, and ensure tests cover what you add or modify.

## Git Workflow

- Commit policy:
  - Only `git add` and `git commit`. Do not push.
  - Keep commits focused and descriptive. Prefer imperative style (e.g., "Add X", "Fix Y").
  - If a change spans multiple logical parts (e.g., Python + JS), either split commits or explain clearly in a single commit message.
- Tags (optional):
  - You may create lightweight or annotated tags to mark milestones, e.g., `v0.1.0-js-port-ready`.
  - Do not push tags; the maintainer will publish tags.
- Branches:
  - Work on the default branch unless a feature branch is explicitly requested.

## Repository Layout

- `python/`
  - `polliLib/` — modular Python package (no server). Entrypoint via `__init__.py`; examples in `__main__.py`.
    - `base.py` — core client, model listing/cache, helpers, 5–8 digit random seeds, URL helpers.
    - `images.py` — image generation, timestamped save, direct fetch.
    - `text.py` — text generation (plain or JSON string parsing).
    - `chat.py` — chat completion, SSE streaming, function calling (tools).
    - `stt.py` — speech‑to‑text via `input_audio` messages.
    - `vision.py` — image analysis for URL and local files.
    - `feeds.py` — public image/text SSE feeds (optional image bytes/data URLs).
  - `requirements.txt` — Python deps (requests, pytest).
  - `README.md` — usage, examples, testing.
  - `tests/` — pytest suite; offline tests using a stubbed `requests.Session`.
- `javascript/`
  - `polliLib/` — modular JavaScript (ESM) library (no server). Single import surface via `index.js`.
    - `base.js` — core client and helpers (uses global `fetch` or injected).
    - `images.js`, `text.js`, `chat.js`, `stt.js`, `vision.js`, `feeds.js` — mirrors Python.
    - `client.js` — composes mixins into `PolliClient`.
  - `package.json` — ESM config; `node --test` test script.
  - `README.md` — usage and testing.
  - `tests/` — uses Node’s `node:test`; offline with stubbed `fetch`.
- `.gitignore` — Python + common dev artifacts; JS has its own inside `javascript/`.

## Core Behaviors (Both Implementations)

- Safe defaults:
  - Random seed is always generated when not provided; must be 5–8 digits.
  - Image defaults: model `flux`, 512×512, `nologo=true`, no `image`, no `referrer`.
  - Text default model: `openai`.
  - Long timeouts for image generation; shorter for text/chat.
- Auth / metadata:
  - Support `referrer` and optional `token` across endpoints (as query params for GET, inside JSON for POST).
- Streaming (SSE):
  - Parse `data:` lines, handle `[DONE]` sentinel, ignore comments/empty lines.
  - Chat streaming yields `delta.content` text; feeds yield parsed JSON.
  - Image feed can optionally attach `image_bytes` or `image_data_url` (base64 with MIME type).

## Adding or Changing Functionality

- Keep the modular structure. Add new features as small mixins/modules aligned with existing patterns.
- Defaults first: prefer non‑breaking additions; maintain the façade API.
- Tests are mandatory for new behavior. Mirror Python and JS tests where feasible.
- If network behavior is needed, add offline tests with fakes/stubs:
  - Python: stub `requests.Session` methods.
  - JS: stub `fetch` and simulate `ReadableStream` for SSE.
- Update the relevant README sections when the public API changes.

## Running & Testing

- Python:
  - Examples: `python -m polliLib` (runs small demos; public feed examples are commented since they are endless).
  - Tests: `pytest python/tests` (offline).
- JavaScript:
  - Library only (no server). Import from `polliLib/index.js`.
  - Tests: `node --test javascript/tests` or `npm run test` (offline).

## Code Style & Conventions

- Keep changes minimal and focused; avoid drive‑by refactors.
- Maintain consistent naming between Python and JS where reasonable (e.g., `generate_text`, `chat_completion_stream`).
- Do not embed secrets. Do not hard‑code credentials.
- Respect timeouts and streaming memory safety (prefer streaming to disk when paths are provided).
- Follow existing commit message tone and clarity.

## Versioning

- The libraries expose `__version__` / `__version__`‑like constants. Increment judiciously when making externally‑visible changes. You may propose tags in commit messages; actual tag creation/publication is handled by the maintainer.

## Your Role

When operating in this repository, you act as the lead developer:
- Make thoughtful architectural decisions that fit the modular pattern.
- Keep APIs stable and documented.
- Ensure every new feature or bugfix is covered by tests in both Python and JavaScript where applicable.
- Favor safe defaults, offline tests, and clarity over cleverness.

If you have questions about priorities or scope, document assumptions in commit messages and, where helpful, in short inline comments or README updates.

