# ComfyUI AI Agent Guide

## Architecture & Flow
- `main.py` bootstraps CLI flags (`comfy/cli_args.py`), applies folder overrides (`folder_paths.py`), executes custom-node prestartup hooks, then spins up `server.PromptServer` and the async `execution.PromptExecutor` worker thread.
- `server.py` owns the aiohttp app, websocket protocol (progress, feature flags, binary previews), file-serving routes, and security middleware (origin checks, CSP, cache control) – respect these patterns when adding endpoints.
- Generations enter `execution.PromptQueue`, which feeds `execution.py` (node scheduler, caching, GC hooks); any node work must be async-safe and interact with `comfy_execution` helpers (`graph`, `caching`, `progress`).
- User data lives under `models/`, `input/`, `output/`, `temp/`, and `user/`; `folder_paths` centralizes discovery and must be updated if you introduce new storage locations.

## Modules & Responsibilities
- `comfy/` holds model logic (loading, quantization, samplers). Use `QUICK_REFERENCE.md` when touching `supported_models.py`, `model_base.py`, or `model_patcher.py` to follow the five supported state-dict customization patterns.
- `comfy_execution/` handles prompt graph parsing/validation; leverage existing `GraphBuilder`, `DynamicPrompt`, and cache abstractions instead of reimplementing scheduling.
- `nodes.py` plus `custom_nodes/` implement the node registry. Custom packages may ship `prestartup_script.py`; never block import-time execution and gate optional features behind CLI flags like `--disable-all-custom-nodes`.
- `app/` contains services (logger, database bootstrap, user/model/subgraph managers) and frontend management; database init happens via `app/database/db.py` and requires SQLAlchemy/Alembic deps from `requirements.txt`.
- `api_server/` is for internal REST glue. Anything under `api_server/routes/internal` is unstable by design – keep integrations behind feature flags or explicit opt-ins.

## Frontend & External Integrations
- The Vue/TS frontend ships as the `comfyui-frontend-package`; `app/frontend_management.py` verifies the installed version and can pull alternates via `--front-end-version owner/repo@tag` or `--front-end-root path`.
- When working on API nodes, follow `comfy_api_nodes/README.md`: run against staging via `python main.py --comfy-api-base https://stagingapi.comfy.org`, keep OpenAPI filtering scripts (`redocly`, `datamodel-codegen`) in sync, and only mark endpoints as Released once deployed.
- WebSocket feature negotiation lives in `comfy_api/feature_flags.py`; add new capabilities there and guard server pushes with `supports_feature(..., "flag")` to stay backward compatible.

## Developer Workflows
- Install deps with `pip install -r requirements.txt` (plus platform-specific torch instructions in `README.md`). Run the server via `python main.py [--listen 0.0.0.0,:: --port 8188 ...]`; AMD/Intel/Ascend switches are already exposed via CLI.
- Automated tests live in two suites: inference regression (`pytest tests/inference` after installing `tests/README.md` extras) and fast unit tests (`pip install -r tests-unit/requirements.txt && pytest tests-unit`).
- Use `--extra-model-paths-config path/to/extra_model_paths.yaml` or edit the root `extra_model_paths.yaml` to share checkpoints; don’t hardcode filesystem paths in code.
- For multi-user or API-disabled deployments, honor `--multi-user`, `--disable-api-nodes`, and other switches – new features should wire through CLI parsing first, then read from `args`.
- Logging is centralized via `app/logger.py`; prefer `app.logger.setup_logger` and per-module `logging.getLogger` instead of print statements (enforced by Ruff `T` rules in `pyproject.toml`).

## Coding Conventions & Pitfalls
- Stick to ASCII in source files, follow the lint rules configured in `pyproject.toml` (`ruff` focuses on Pyflakes/S* rules; pylint is tolerant but still enforces basics like valid first arg names).
- Node implementations should subclass `_ComfyNodeInternal` when possible to get hidden inputs (`io.Hidden.*`), list mapping, and thread-safe execution wrappers for free.
- VRAM/offload behaviour is controlled by `comfy/model_management.py`; respect existing helpers (e.g., `soft_empty_cache`, `cuda_malloc` guard rails) when adding memory-intensive code.
- Any HTTP file access must route through the helpers in `folder_paths`/`server` (e.g., `get_dir_by_type`, duplicate hashing) to preserve sandboxing and deduplication semantics.
- Long-running background work (downloads, cache rebuilds) should integrate with the asyncio loop returned by `start_comfyui()` rather than spawning unmanaged threads.
