<p align="center">
  <img src="assets/logo.svg" alt="PromptMill" width="300">
</p>

<p align="center">
  <strong>AI-powered prompt generator for video, image, and creative content</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-targets">Targets</a> •
  <a href="#configuration">Configuration</a>
</p>

---

A self-contained web UI with selectable LLMs (scaled by GPU VRAM) for generating optimized prompts. No external API dependencies - runs entirely locally.

## Features

- **14 Preset Roles** for video, image, and creative tasks
- **7 LLM Options** from 0.5B to 14B parameters
- **Auto-detects GPU VRAM** and selects optimal model
- Dark mode UI with streaming text generation
- One-click copy button

## Quick Start

### Docker (Recommended)

```bash
# GPU (NVIDIA CUDA) - auto-detects VRAM and selects best model
docker compose --profile gpu up -d

# CPU only
docker compose --profile cpu up -d
```

Open http://localhost:7610

Models auto-download on first use and persist in `./models/`.

### Manual Setup

```bash
# GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install gradio huggingface_hub
python app.py

# CPU Only
pip install llama-cpp-python gradio huggingface_hub
python app.py
```

## Supported Targets

### Video Generation
- **Wan2.1** - Open-source text-to-video with cinematic quality
- **Wan2.2** - Latest Wan with improved motion and physics
- **Hunyuan Video** - Tencent open-source text-to-video
- **Hunyuan Video 1.5** - Extended duration and image-to-video

### Image Generation
- **Stable Diffusion** - SD/SDXL prompts with quality tags
- **Midjourney** - Artistic MJ-style prompts with parameters
- **FLUX** - Black Forest Labs natural language prompts
- **DALL-E 3** - OpenAI detailed scene descriptions
- **ComfyUI** - Workflow-compatible positive/negative prompts

### Creative & Coding
- **Story Writer** - Creative writing and narratives
- **Code Generator** - Programming snippets with best practices
- **Technical Writer** - Documentation and READMEs
- **Marketing Copy** - Ad copy, social media, CTAs
- **SEO Content** - Blog posts with meta optimization

## LLM Options

| GPU VRAM | Model | Parameters | Quality |
|----------|-------|------------|---------|
| CPU Only | Qwen2.5 0.5B Q8 | 0.5B | Basic |
| 4GB | Qwen2.5 1.5B Q4_K_M | 1.5B | Good |
| 6GB | Qwen2.5 3B Q4_K_M | 3B | Great |
| 8GB | Dolphin Mistral 7B Q4_K_M | 7B | Excellent |
| 12GB | Qwen2.5 7B Q6_K_L | 7B | High |
| 16GB+ | Qwen2.5 14B Q4_K_M | 14B | Premium |
| 24GB+ | Qwen2.5 14B Q8 | 14B | Maximum |

## Configuration

The app auto-detects GPU VRAM and selects the optimal model:
- GPU detected: Uses all GPU layers, selects model by VRAM
- No GPU: Uses CPU mode with lightweight 0.5B model

You can manually override the model and GPU layers in the UI.

## File Structure

```
.
├── app.py              # Main application
├── assets/
│   └── logo.svg        # PromptMill logo
├── Dockerfile.gpu      # GPU/CUDA build
├── Dockerfile.cpu      # CPU-only build
├── docker-compose.yml  # Docker profiles
├── requirements.txt
└── models/             # Downloaded models (persisted)
```

## License

MIT
