# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PromptMill** (v3.1.0) - A self-contained Gradio web UI with **selectable LLMs based on GPU VRAM** for generating optimized prompts for:
- **Video** (31 targets): Wan2.1, Wan2.2, Wan2.5, Hunyuan Video, Hunyuan Video 1.5, Runway Gen-3, Runway Gen-4.5, Kling AI, Kling 2.1, Kling 2.5, Kling Video O1, Pika Labs, Pika 2.1, Pika 2.2, Luma Dream Machine, Luma Ray2, Sora, Sora 2, Veo, Veo 3, Veo 3.1, Hailuo AI, Seedance, SkyReels V1, Mochi 1, CogVideoX, LTX Video, Open-Sora, MovieGen, Pyramid Flow, Allegro
- **Image** (31 targets): Stable Diffusion, SD 3.5, Midjourney, Midjourney v7, FLUX, FLUX 2, FLUX Pro, FLUX 2 Max, DALL-E 3, ComfyUI, Ideogram, Ideogram 3, Leonardo AI, Adobe Firefly, Adobe Firefly 3, Recraft, Recraft V3, Imagen 3, Imagen 4, GPT-4o Images, GPT Image 1.5, Reve Image, HiDream-I1, Qwen-Image, FLUX Kontext, Grok Image, Hunyuan Image 3.0, Seedream 4.5, Gemini 3 Pro Image, Playground v3, Krea AI
- **Audio** (18 targets): Suno AI, Suno v4.5, Suno v5, Udio, Udio 2.0, ElevenLabs, Eleven Music, Mureka AI, SOUNDRAW, Beatoven.ai, Stable Audio 2.0, MusicGen, ACE Studio, AIVA, Boomy, Google MusicFX, Riffusion, Bark
- **3D** (18 targets): Meshy, Meshy 4, Tripo AI, Tripo 2.0, Rodin, Rodin Gen-2, Spline, Sloyd, 3DFY.ai, Luma Genie, Masterpiece X, Hunyuan3D, Trellis, TripoSR, Unique3D, SF3D, InstantMesh, CSM 3D
- **Creative** (34 targets): Story, code, technical docs, marketing, SEO, screenplays, social media, podcasts, UX, press releases, poetry, data analysis, business plans, academic writing, tutorials, newsletters, legal docs, grant writing, API documentation, courses, pitch decks, meeting notes, changelogs, recipes, travel guides, workout plans

**Total: 132 specialized prompt templates**

## Commands

### Docker (Recommended)

```bash
# GPU (NVIDIA CUDA) - auto-detects VRAM and selects best model
docker compose --profile gpu up -d

# CPU only
docker compose --profile cpu up -d
```

### Manual Setup

```bash
# GPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install gradio huggingface_hub
python -m promptmill

# CPU
pip install llama-cpp-python gradio huggingface_hub
python -m promptmill

# Run tests
PYTHONPATH=src pytest tests/unit -v
```

Access at http://localhost:7610

## Architecture

**Hexagonal Architecture** (Ports and Adapters) with **Domain-Driven Design** using pure Python 3.12+.

### Layer Overview

```
src/promptmill/
├── domain/           # Pure business logic (no external dependencies)
├── application/      # Use cases and services
├── infrastructure/   # External adapters and config
└── presentation/     # Gradio UI
```

### Domain Layer (`src/promptmill/domain/`)

Pure Python with no external dependencies.

**Entities:**
- `Model` - LLM model configuration (frozen dataclass)
- `Role` - Prompt template with category (frozen dataclass)
- `GPUInfo` - GPU detection result (value object)

**Value Objects:**
- `PromptGenerationRequest` - Validated input (user_input, role_name, temperature, max_tokens)
- `PromptGenerationResult` - Generation output with metadata

**Ports (Interfaces):**
- `LLMPort` - Abstract LLM operations (generate, load, unload)
- `ModelRepositoryPort` - Model storage and download
- `GPUDetectorPort` - GPU detection
- `RoleRepositoryPort` - Role/template retrieval

**Exceptions:**
- `DomainError` - Base exception
- `ModelNotLoadedError`, `ModelLoadError`, `ModelDownloadError`
- `RoleNotFoundError`, `InvalidTemperatureError`, etc.

### Application Layer (`src/promptmill/application/`)

**Use Cases:**
- `GeneratePromptUseCase` - Generate prompt with streaming
- `LoadModelUseCase` - Load LLM with model switching
- `UnloadModelUseCase` - Unload and free memory
- `SelectModelByVRAMUseCase` - Auto-select model by GPU VRAM
- `DeleteModelUseCase` - Delete downloaded model
- `GetHealthStatusUseCase` - Health check

**Services:**
- `PromptService` - Coordinates generation with auto-unload timer
- `ModelService` - Model lifecycle management
- `HealthService` - Health status aggregation

### Infrastructure Layer (`src/promptmill/infrastructure/`)

**Adapters:**
- `LlamaCppAdapter` - Implements `LLMPort` using llama-cpp-python
- `HuggingFaceAdapter` - Implements `ModelRepositoryPort` using huggingface_hub
- `NvidiaSmiAdapter` - Implements `GPUDetectorPort` using nvidia-smi
- `RoleRepositoryAdapter` - Implements `RoleRepositoryPort` from static data

**Configuration:**
- `Settings` - Environment-based configuration
- `MODEL_CONFIGS` - 7 model configurations keyed by VRAM tier
- `ROLES_DATA` - 132 role definitions

### Presentation Layer (`src/promptmill/presentation/`)

- `GradioApp` - Gradio Blocks interface with event handlers
- `PromptMillTheme` - Custom dark theme

### Dependency Injection (`src/promptmill/container.py`)

Manual DI container using `@property` decorators with lazy initialization:

```python
@dataclass
class Container:
    settings: Settings

    @property
    def llm(self) -> LLMPort:
        return self._llm  # Lazy singleton

    @property
    def gradio_app(self) -> GradioApp:
        # Wires all dependencies
```

### Entry Point (`src/promptmill/__main__.py`)

```python
def main() -> None:
    settings = Settings.from_environment()
    container = Container(settings=settings)
    app = container.gradio_app
    app.create()
    app.launch(host=settings.host, port=settings.port)
```

## Key Patterns

- **Hexagonal Architecture**: Domain isolated from infrastructure
- **Dependency Injection**: Constructor injection, no framework
- **Immutable Data**: `frozen=True, slots=True` dataclasses
- **Python 3.12 Features**: `@override`, `StrEnum`, pattern matching, type aliases
- **Lazy Loading**: Heavy dependencies (llama_cpp, huggingface_hub) imported only when needed
- **Thread Safety**: RLock in PromptService for timer management
- **Auto-Cleanup**: Model unloads after 10 seconds of inactivity
- **Streaming**: Generator-based prompt generation

## Model Configuration

7 uncensored Dolphin LLM options scaled by VRAM (1B to 8B parameters):

| VRAM | Model Key | Context |
|------|-----------|---------|
| CPU  | `cpu_only` | 4096 |
| 4GB  | `4gb_vram` | 4096 |
| 6GB  | `6gb_vram` | 4096 |
| 8GB  | `8gb_vram` | 4096 |
| 12GB | `12gb_vram` | 4096 |
| 16GB+ | `16gb_vram` | 4096 |
| 24GB+ | `24gb_vram` | 131072 |

## Testing

Tests organized by layer in `tests/unit/`:

```bash
# Run all unit tests
PYTHONPATH=src pytest tests/unit -v

# Run specific layer
PYTHONPATH=src pytest tests/unit/domain -v
PYTHONPATH=src pytest tests/unit/application -v
PYTHONPATH=src pytest tests/unit/infrastructure -v
```

## File Structure

```
PromptMill/
├── src/promptmill/
│   ├── __init__.py              # Package + version
│   ├── __main__.py              # Entry point
│   ├── container.py             # DI container
│   ├── domain/
│   │   ├── entities/            # Model, Role, GPUInfo
│   │   ├── value_objects/       # Request/Result VOs
│   │   ├── ports/               # Abstract interfaces
│   │   └── exceptions.py        # Domain exceptions
│   ├── application/
│   │   ├── use_cases/           # 6 use cases
│   │   └── services/            # 3 services
│   ├── infrastructure/
│   │   ├── adapters/            # 4 adapters
│   │   ├── config/              # Settings, ModelConfigs
│   │   └── persistence/         # RolesData (132 roles)
│   └── presentation/
│       ├── gradio_app.py        # Gradio UI
│       └── theme.py             # Dark theme
├── tests/
│   ├── conftest.py              # Shared fixtures
│   └── unit/                    # Layer-specific tests
├── pyproject.toml               # Python 3.12+, src layout
├── Dockerfile.cpu               # CPU build
├── Dockerfile.gpu               # CUDA build
├── docker-compose.yml           # Docker orchestration
└── models/                      # Downloaded LLMs
```

## Code Quality

- **Python**: 3.12+ required (uses `@override`, `StrEnum`, etc.)
- **Linting**: Ruff with strict rules targeting `py312`
- **Type Hints**: Full type annotations throughout
- **Testing**: pytest with 74+ unit tests
- **Architecture**: Clean separation, no circular dependencies
