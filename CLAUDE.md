# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**PromptMill** (v3.3.0) - a self-contained Gradio web UI plus REST API that generates
optimized prompts for image, video, audio, 3D and creative-writing targets. 146 prompt
templates across those five categories; the model that writes them is picked automatically
from the GPU's VRAM.

The canonical lists live in code, not here:

- targets/templates: `src/promptmill/infrastructure/persistence/` (`ROLES_DATA`)
- model tiers: `src/promptmill/infrastructure/config/` (`MODEL_CONFIGS`)
- routes/wiring: `src/promptmill/container.py`
- HTTP surface: `src/promptmill/presentation/api.py`

## Commands

```bash
./start.sh                              # detects GPU, pulls the GHCR image, waits for health
docker compose --profile gpu up -d      # NVIDIA CUDA, prebuilt image
docker compose --profile cpu up -d      # CPU only, prebuilt image

# Lint, format check and the full suite. No llama-cpp compile, no GPU.
docker compose -f docker-compose.yml -f docker-compose.build.yml \
  --profile test run --rm test

# Build from source instead of pulling
docker compose -f docker-compose.yml -f docker-compose.build.yml \
  --profile gpu up -d --build
```

UI on http://localhost:7610, OpenAPI docs on /docs. `PYTHONPATH=src` is required for any
direct invocation, the package uses a src layout.

`docker-compose.yml` only references the published images. Anything that builds lives in
`docker-compose.build.yml`, so a third party never triggers a CUDA compile by accident.

## Architecture

Hexagonal (ports and adapters) with DDD, pure Python 3.13+.

```
src/promptmill/
├── domain/           # entities, value objects, ports, exceptions - NO external imports
├── application/      # use cases + services
├── infrastructure/   # adapters (llama-cpp, huggingface_hub, nvidia-smi), config, roles data
└── presentation/     # Gradio UI, REST API, examples, session history, theme
```

**The domain layer imports nothing external.** That is the rule the whole structure exists
for: adapters implement the ports (`LLMPort`, `ModelRepositoryPort`, `GPUDetectorPort`,
`RoleRepositoryPort`), the domain only ever sees the port.

`container.py` is a hand-written DI container: `@property` per dependency, lazy singletons,
no framework. Adding a dependency means adding a property there and nothing else.

## Behavior worth knowing before changing it

- **Models load lazily and unload themselves.** `PromptService` drops the model after
  10 seconds of inactivity (RLock-guarded timer). A test that seems to hang on a loaded
  model is usually waiting on that timer.
- **Heavy imports are deferred.** `llama_cpp` and `huggingface_hub` are imported inside the
  adapters, not at module scope, so the domain and the tests import fast. Keep it that way.
  The test image deliberately omits `llama-cpp-python` for the same reason.
- **Generation is a generator** (streamed into Gradio and into `/api/generate/stream`), not
  a return value.
- **"Already loaded" means path AND GPU split.** The GPU-layers slider is a real override;
  `LoadModelUseCase` reloads when it changes, so a path-only comparison would silently
  ignore the user.
- **`chat_format` belongs to the `Model`,** not the adapter. `Settings.default_chat_format`
  is only the fallback. A wrong template produces garbage silently, never a crash.
- **Context grows with the tier** (4K up to 32K on 24GB). `vram_required` is weights plus
  KV cache at that context, so it must be recomputed when a context length changes.
- Model selection is by VRAM tier, 7 uncensored Dolphin 3.0 variants from 1B to 8B.
- **UI state is per session.** Examples and history live in `gr.State`, never in the
  `GradioApp` instance, which is shared across all browsers.
- **The CPU image installs llama-cpp-python from the project's own wheel index,**
  `--only-binary`. A PyPI source build of 0.3.35 links its ggml shared objects without
  libstdc++, so they fail to `dlopen` with an undefined C++ ABI symbol. The GPU image
  builds from source (CUDA 13 has no published wheel) and passes `-lstdc++` explicitly.
- **`libgomp1` is required in the GPU runtime stage.** The CUDA runtime base image has no
  OpenMP runtime, but ggml links against it.
- **A broken inference runtime is invisible until generation.** `llama_cpp` is imported
  lazily and loaded through ctypes, so the container starts healthy and only fails on the
  first prompt. CI imports it explicitly for that reason.

## Conventions

- Python 3.13+ features are used deliberately (`@override`, `StrEnum`, `type` aliases,
  pattern matching).
- Dataclasses are `frozen=True, slots=True`. Keep new value objects immutable.
- Ruff with strict rules, target `py313`.
- Tests mirror the layers: `tests/unit/{domain,application,infrastructure,presentation}`
  plus `tests/integration`.
- **Never assert a target or role count as a literal.** Derive it from `ROLES_DATA`, or
  adding one template breaks unrelated tests.
