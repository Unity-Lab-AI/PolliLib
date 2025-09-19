# Contributing to PolliLib

Thanks for your interest in improving PolliLib! This guide covers how we work across Python and JavaScript, how to update the API AST, and the repository norms.

## Ground Rules
- Stage and commit only. Do not push from automation or local scripts. Maintainers manage the remote and will push.
- Keep Python and JavaScript APIs in sync. If you change parameters, defaults, return types, or behavior in one language, reflect it in the other and in the AST.
- Add/update tests for any feature or bug fix in both `python/tests` and `javascript/tests` where applicable.
- Keep changes minimal and focused. Avoid drive‑by refactors that aren’t related to the task.

## Workflow
1) Open an issue or pick an existing one.
2) Implement the change in Python and/or JavaScript.
3) Update `/AST` to reflect any API surface changes (names, params/defaults, returns, streaming contracts, errors, and cross‑language naming).
4) Update the language READMEs and relevant sections of the root README.
5) Run tests locally:
   - Python: `python -m pip install -r python/requirements.txt && pytest python/tests`
   - JS: `node --test javascript/tests`
6) Stage and commit your changes (no push). Optionally add a tag if requested by maintainers.

## Coding Conventions
- Python: follow existing style in `python/polliLib` (snake_case, explicit types where helpful, keep modules small and composable).
- JavaScript: ESM modules, camelCase options, avoid server code, library‑only. Support Node 18+ with global `fetch`; allow injected `fetch` for other environments.
- Defaults: Maintain consistent defaults across languages (e.g., image defaults, random seed generation, timeout units).
- Error handling: Raise/throw helpful errors for invalid inputs; prefer explicit checks.

## AST Maintenance
- Files: `/AST/*.ast.json` and `/AST/polli.ast.json`.
- Purpose: single source of truth for public API across languages.
- Required updates: whenever you change a public surface or behavior.
- Include: parameters and defaults, return and event shapes, SSE contracts (terminators, payload lines), error surfaces, and cross‑language naming/units.

## Tests
- Expand unit tests for new code paths and edge cases.
- Stream handling: prefer deterministic, bounded tests. For long‑running streams, keep examples commented by default.

## Commit Messages
- Use succinct, descriptive messages (Conventional Commit flavor is welcome):
  - `feat(images): add include_data_url option`
  - `fix(chat): handle nested tool arguments`
  - `docs(ast): update text defaults`
  - `chore: ignore .tools and tmp-*`

## Security
- Do not commit secrets. If you need to use `referrer` or `token`, pass them via parameters or environment during testing.

## Need Help?
- Open an issue and describe your use case or problem clearly, including the language(s) and subsystem(s) involved (images, text, chat, vision, stt, feeds).
