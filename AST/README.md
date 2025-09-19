AST Reference for PolliLib (Python and JavaScript)

Overview
- This folder contains a language‑agnostic API AST describing the PolliLib client across Python and JavaScript.
- Use it as a stable reference when evolving APIs and for porting features between languages.
- The AST captures: modules, entities (classes/mixins), method signatures, parameters and defaults, return types, error surfaces, HTTP shapes, streaming/event contracts, and language-specific naming mappings.

Scope and Principles
- Single source of truth: update the AST whenever any public API shape changes in python/ or javascript/.
- Language neutral: describe behaviors and types in a way that applies to both Python and JS.
- Precise defaults: explicitly record default values (e.g., seed is randomly generated with 5–8 digits when omitted).
- Minimal coupling: keep per-module ASTs focused; shared types live in the root polli.ast.json under types.

Files
- polli.ast.json: Top-level manifest, shared types, module index, and cross-language conventions.
- base.ast.json: Model listing and base client helpers (URLs, seed).
- images.ast.json: Image generation and image fetching helpers.
- text.ast.json: Text generation.
- chat.ast.json: Chat completion, streaming, and tool/function-calling.
- vision.ast.json: Vision (image URL/file analysis).
- stt.ast.json: Speech-to-text (transcribe_audio with input_audio content).
- feeds.ast.json: Public SSE feeds for images and text.
- client.ast.json: Composition of mixins into the PolliClient façade.

Updating the AST
1) Make your code changes in python/ or javascript/.
2) Diff the public surface: function names, parameters, defaults, return types, and behaviors.
3) Update the corresponding module .ast.json file(s):
   - Reflect new/changed parameters (including default values and nullability).
   - Adjust return types and streaming/event contracts.
   - Record any new error conditions.
   - Note language-specific option names (snake_case in Python, camelCase in JS).
4) If new shared types are introduced, update polli.ast.json under types.
5) Validate the mapping: ensure python ↔ js functions exist and semantics match.
6) Run tests in both implementations to confirm behavior:
   - Python: `python -m pip install -r python/requirements.txt && pytest python/tests`
   - JavaScript: `node --test javascript/tests`
7) Stage and commit your AST changes (see AGENTS.md: stage+commit only; do not push here).

Conventions and Naming
- Python options use snake_case (e.g., out_path, timeout), JavaScript uses camelCase (outPath, timeoutMs).
- When a parameter is optional, null/None omits it from requests.
- Seeds: if omitted, a random 5–8 digit integer is generated in both implementations.
- Timeouts: Python in seconds (float), JS in milliseconds (number). The AST records both and their defaults.

Quality Checklist for AST Updates
- Defaults are explicitly documented.
- HTTP endpoints and methods are correct.
- All query/body parameters are listed with correct names.
- Streaming contracts include termination signals (e.g., [DONE]) and event payload shapes.
- Return union types and side effects (e.g., writing files) are captured.
- Error surfaces reflect common runtime exceptions.

Governance
- The AST mirrors the libraries at version noted in polli.ast.json. Increment that version when any breaking change to the API is recorded.
- Keep the AST in sync with READMEs in python/ and javascript/.

