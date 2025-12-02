# AI Prompt Generator

A self-contained web UI for generating optimized prompts for AI video, image, and creative tasks. Includes a built-in uncensored LLM - no external dependencies required.

## Features

- **12 Preset Roles** for different AI models and tasks
- Built-in Dolphin-Mistral-7B model (uncensored, creative)
- Auto-detects GPU, falls back to CPU if unavailable
- Streaming text generation
- One-click copy button

## Supported Roles

### Video Generation
- **Wan2.1** - Alibaba text-to-video
- **Hunyuan Video** - Tencent text-to-video

### Image Generation
- **Stable Diffusion** - SD/SDXL prompts
- **Midjourney** - Artistic MJ-style prompts
- **FLUX** - Black Forest Labs prompts
- **DALL-E 3** - OpenAI image prompts
- **ComfyUI** - Workflow-compatible prompts

### Creative & Coding
- **Story Writer** - Creative writing
- **Code Generator** - Programming snippets
- **Technical Writer** - Documentation/README
- **Marketing Copy** - Ad copy and social
- **SEO Content** - Blog posts and articles

## Quick Start

### Default (CPU)

```bash
docker compose up -d
```

### With GPU (NVIDIA)

```bash
docker compose --profile gpu up -d
```

Open http://localhost:7610

Model auto-downloads on first run (~4GB) and persists in `./models/`.

## Manual Setup

### GPU (CUDA)

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install gradio huggingface_hub
python app.py
```

### CPU Only

```bash
pip install llama-cpp-python gradio huggingface_hub
python app.py
```

## Model Info

- **Model:** Dolphin 2.6 Mistral 7B (Q4_K_M)
- **Size:** ~4.4GB
- **Type:** Uncensored, instruction-tuned

## Configuration

The app auto-detects GPU availability on startup:
- GPU detected: Uses all GPU layers (-1)
- No GPU: Uses CPU only (0)

You can adjust GPU layers in the UI settings panel.

## File Structure

```
.
├── app.py              # Main application
├── Dockerfile.gpu      # GPU/CUDA build
├── Dockerfile.cpu      # CPU-only build
├── docker-compose.yml  # Docker profiles
├── requirements.txt
└── models/             # Downloaded model (persisted)
```
