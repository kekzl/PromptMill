# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PromptMill** (v2.3.0) - A self-contained Gradio web UI with **selectable LLMs based on GPU VRAM** for generating optimized prompts for:
- **Video** (22 targets): Wan2.1, Wan2.2, Wan2.5, Hunyuan, Hunyuan 1.5, Runway Gen-3, Kling, Kling 2.1, Pika, Pika 2.1, Luma Dream Machine, Luma Ray2, Sora, Veo, Veo 3, Hailuo AI (MiniMax), Seedance, SkyReels V1, Mochi 1, CogVideoX, LTX Video, Open-Sora
- **Image** (21 targets): Stable Diffusion, SD 3.5, Midjourney, FLUX, FLUX 2, DALL-E, ComfyUI, Ideogram, Leonardo, Firefly, Recraft, Imagen 3, Imagen 4, GPT-4o Images, Reve Image, HiDream-I1, Qwen-Image, Recraft V3, FLUX Kontext, Ideogram 3, Grok Image
- **Audio** (13 targets): Suno AI, Udio, ElevenLabs, Eleven Music, Mureka AI, SOUNDRAW, Beatoven.ai, Stable Audio 2.0, MusicGen, Suno v4.5, ACE Studio, AIVA, Boomy
- **3D** (12 targets): Meshy, Tripo AI, Rodin, Spline, Sloyd, 3DFY.ai, Luma Genie, Masterpiece X, Hunyuan3D, Trellis, TripoSR, Unique3D
- **Creative** (34 targets): Story, code, technical docs, marketing, SEO, screenplays, social media, podcasts, UX, press releases, poetry, data analysis, business plans, academic writing, tutorials, newsletters, legal docs, grant writing, API documentation, courses, pitch decks, meeting notes, changelogs, recipes, travel guides, workout plans

**Total: 102 specialized prompt templates**

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
python app.py

# CPU
pip install llama-cpp-python gradio huggingface_hub
python app.py

# Run tests
pip install pytest pytest-cov
pytest
```

Access at http://localhost:7610

## Architecture

Single-file application (`app.py`) with:

### Configuration (Lines 46-80)
- `__version__`: Current version string
- `logger`: Structured logging with timestamps
- Environment variables: `SERVER_HOST`, `SERVER_PORT`, `MODELS_DIR`
- Constants: `MAX_PROMPT_LENGTH`, `MIN/MAX_TEMPERATURE`, `MIN/MAX_TOKENS`

### Model System (Lines 85-145)
- `MODEL_CONFIGS`: 7 uncensored Dolphin LLM options scaled by VRAM (1B to 8B parameters)
  - CPU Only: Dolphin 3.0 Llama 3.2 1B Q8
  - 4GB VRAM: Dolphin 3.0 Llama 3.2 3B Q4_K_M
  - 6GB VRAM: Dolphin 3.0 Llama 3.2 3B Q8
  - 8GB VRAM: Dolphin 3.0 Llama 3.1 8B Q4_K_M (default)
  - 12GB VRAM: Dolphin 3.0 Llama 3.1 8B Q6_K_L
  - 16GB+ VRAM: Dolphin 3.0 Llama 3.1 8B Q8
  - 24GB+ VRAM: Dolphin 2.9.4 Llama 3.1 8B Q8 (131K context)
- `model_lock`: Threading lock for concurrent request safety
- Global state: `llm`, `current_model_key`, `unload_timer`

### Core Functions
- `detect_gpu()`: NVIDIA GPU detection via `nvidia-smi` with proper error handling
- `load_model()`: Thread-safe lazy loading with model switching
- `unload_model()`: Thread-safe memory cleanup with garbage collection
- `generate_prompt()`: Streaming generation with input validation
- `schedule_unload()`: Auto-unload timer (10 seconds of inactivity)

### UI Components
- `create_theme()`: Custom dark mode Gradio theme
- `create_ui()`: Gradio Blocks interface with role/model selection
- `get_logo_html()`: Logo loading from assets/logo.svg

## Key Patterns

- **Thread Safety**: `model_lock` protects global model state for concurrent requests
- **Structured Logging**: All operations logged via Python's `logging` module
- **Input Validation**: Prompt length, temperature, and token limits enforced
- **Error Handling**: Specific exception types caught with descriptive messages
- **Lazy Loading**: Heavy dependencies imported only when needed
- **Auto-Cleanup**: Model unloads after 10 seconds to free VRAM
- **Role System**: Format `[Category] RoleName` parsed by `parse_role_choice()`

## Testing

Tests are in `tests/test_app.py`:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test class
pytest tests/test_app.py::TestGPUDetection
```

## File Structure

```
PromptMill/
├── app.py              # Main application (~5000 lines)
├── pyproject.toml      # Project config & dependencies
├── README.md           # User documentation
├── CLAUDE.md           # Developer guidance
├── tests/
│   ├── __init__.py
│   └── test_app.py     # Unit tests
├── assets/
│   └── logo.svg        # Application logo
├── Dockerfile.gpu      # CUDA build
├── Dockerfile.cpu      # CPU build
├── docker-compose.yml  # Docker orchestration
└── models/             # Downloaded LLMs (persisted)
```

## Code Quality

- **Linting**: Ruff with strict rules (E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, PL, RUF)
- **Type Hints**: All functions have type annotations
- **Docstrings**: Google-style docstrings on public functions
- **Testing**: pytest with coverage reporting
