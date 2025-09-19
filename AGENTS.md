# PolliLib Agent Guide

This document explains how agents (and contributors) should work in this repository. It covers workflow expectations, repo structure, code conventions, testing, and how to extend functionality safely. Follow this guide whenever you make changes here.

## Prime Directive

- Stage, commit, and push when ready. Pushing to GitHub is permitted for this project as long as authentication is configured (see “Pushing to GitHub”). You may also create tags for releases.
- You are expected to act as the full lead developer when working in this codebase: make cohesive, well‑documented changes, keep the surface area stable, and ensure tests cover what you add or modify.

## Git Workflow

- Commit policy:
  - Use `git add` + `git commit` with focused, descriptive messages (imperative style: "Add X", "Fix Y").
  - If a change spans multiple logical parts (e.g., Python + JS), split into logical commits or clearly explain scope in one message.
- Tags and releases:
  - Follow Semantic Versioning (MAJOR.MINOR.PATCH), e.g., `0.0.1`.
  - Current project release: `1.0.0`.
  - Create annotated tags for releases, e.g., `git tag -a v1.0.0 -m "v1.0.0"` and push tags when publishing.
- Branches:
  - Work on the default branch unless a feature branch is explicitly requested.

### Pushing to GitHub
- Prerequisite: An accessible `GITHUB_TOKEN` must be present in the host machine’s environment (User or System environment variable).
- Verify remote:
  - `git remote -v` should show the GitHub repository (e.g., `https://github.com/Unity-Lab-AI/PolliLib.git`).
  - If you need to set the remote with a token, you can use one of the following patterns:
    - Unix shells: `git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/Unity-Lab-AI/PolliLib.git"`
    - PowerShell: `git remote set-url origin "https://x-access-token:$env:GITHUB_TOKEN@github.com/Unity-Lab-AI/PolliLib.git"`
- Push commits and tags:
  - `git push origin main`
  - `git push --tags`
Notes:
- Never echo or print the token. Avoid pasting tokens into logs or commit messages.
- Standard Git credential helpers may also source `GITHUB_TOKEN`; if they are configured, a plain `git push` will work once the environment is present.

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

- Use Semantic Versioning (https://semver.org): MAJOR.MINOR.PATCH.
- Current release: `1.0.0`.
- When releasing:
  - Bump versions in both languages (Python `polliLib/__init__.py` `__version__`, JS `javascript/polliLib/index.js` `__version__`).
  - Update `/AST/polli.ast.json` `library_version` if the public API changes.
  - Commit, tag (e.g., `v1.0.0`), and push both commits and tags.

## Your Role

When operating in this repository, you act as the lead developer:
- Make thoughtful architectural decisions that fit the modular pattern.
- Keep APIs stable and documented.
- Ensure every new feature or bugfix is covered by tests in both Python and JavaScript where applicable.
- Favor safe defaults, offline tests, and clarity over cleverness.

If you have questions about priorities or scope, document assumptions in commit messages and, where helpful, in short inline comments or README updates.
