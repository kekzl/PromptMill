# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Prompt Generator - A self-contained Gradio web UI with built-in Dolphin-Mistral-7B LLM for generating optimized prompts for AI video generation (Wan2.1, Hunyuan), image generation (SD, Midjourney, FLUX, DALL-E 3, ComfyUI), and creative tasks.

## Commands

### Docker (Recommended)

```bash
# Default (CPU)
docker compose up -d

# GPU (NVIDIA CUDA)
docker compose --profile gpu up -d
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

- **Model Management**: Auto-downloads Dolphin 2.6 Mistral 7B (Q4_K_M, ~4GB) from HuggingFace on first run to `./models/`
- **GPU Detection**: `detect_gpu()` checks for NVIDIA GPU via `nvidia-smi`, sets `n_gpu_layers` accordingly (-1 for all GPU, 0 for CPU)
- **Role System**: `ROLES` dict defines 12 presets with category, name, description, and system_prompt for each AI model/task type
- **Streaming Generation**: `generate_prompt()` yields tokens progressively using llama-cpp-python's chat completion API
- **UI**: Gradio Blocks interface with role dropdown, settings panel (temperature, max_tokens, GPU layers)

## Key Patterns

- Role selection uses format `[Category] RoleName` parsed by `parse_role_choice()`
- Model loaded lazily on first generation, cached globally in `llm`
- System prompts are role-specific; each role has detailed instructions for its target AI model
