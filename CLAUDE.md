# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PromptMill** - A self-contained Gradio web UI with **selectable LLMs based on GPU VRAM** for generating optimized prompts for:
- **Video**: Wan2.1, Wan2.2, Hunyuan Video, Hunyuan Video 1.5
- **Image**: Stable Diffusion, Midjourney, FLUX, DALL-E 3, ComfyUI
- **Creative**: Story writing, code generation, technical docs, marketing, SEO

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
```

Access at http://localhost:7610

## Architecture

Single-file application (`app.py`) with:

- **Model Selector**: `MODEL_CONFIGS` dict with 7 LLM options scaled by VRAM (0.5B to 14B parameters)
  - CPU Only: Qwen2.5 0.5B Q8
  - 4GB VRAM: Qwen2.5 1.5B Q4_K_M
  - 6GB VRAM: Qwen2.5 3B Q4_K_M
  - 8GB VRAM: Dolphin Mistral 7B Q4_K_M (default)
  - 12GB VRAM: Qwen2.5 7B Q6_K_L
  - 16GB+ VRAM: Qwen2.5 14B Q4_K_M
  - 24GB+ VRAM: Qwen2.5 14B Q8
- **Theme**: Dark mode with custom Gradio theme (`create_theme()`)
- **Branding**: Logo in `assets/logo.svg`, loaded via `get_logo_html()`
- **Model Management**: `load_model()` handles lazy loading, `unload_model()` frees memory when switching models
- **Model Cleanup**: `get_downloaded_models()`, `delete_model()`, `delete_all_models()` for disk space management
- **GPU Detection**: `detect_gpu()` checks for NVIDIA GPU via `nvidia-smi`, sets `n_gpu_layers` accordingly
- **Role System**: `ROLES` dict defines 14 presets for video, image, and creative tasks
- **Streaming Generation**: `generate_prompt()` yields tokens progressively using llama-cpp-python's chat completion API
- **UI**: Gradio Blocks interface with target model dropdown, LLM selector, and generation settings

## Key Patterns

- Role selection uses format `[Category] RoleName` parsed by `parse_role_choice()`
- Model loaded lazily on first generation, cached globally in `llm` with `current_model_key` tracking
- Switching models triggers `unload_model()` to free VRAM before loading new model
- System prompts are role-specific; each role has detailed instructions for its target AI model
- Auto-unload timer (`schedule_unload()`) frees model after 10 seconds of inactivity
