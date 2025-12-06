<div align="center">

<img src="assets/logo.svg" alt="PromptMill" width="320">

<br/>

**AI-powered prompt generator for video, image, and creative content**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF6F00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Ruff](https://img.shields.io/badge/Ruff-Linted-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/kekzl/PromptMill?style=flat-square&logo=github)](https://github.com/kekzl/PromptMill)

[Features](#-features) · [Quick Start](#-quick-start) · [Supported Targets](#-supported-targets) · [Models](#-llm-options) · [Configuration](#%EF%B8%8F-configuration)

</div>

---

## Overview

PromptMill is a self-contained web UI that runs **entirely locally** - no API keys, no cloud dependencies. It uses selectable LLMs (scaled by your GPU VRAM) to generate optimized prompts for the latest AI video and image generators.

<div align="center">
<table>
<tr>
<td align="center"><b>14</b><br><sub>Preset Roles</sub></td>
<td align="center"><b>7</b><br><sub>LLM Options</sub></td>
<td align="center"><b>0.5B-14B</b><br><sub>Parameters</sub></td>
<td align="center"><b>100%</b><br><sub>Local</sub></td>
</tr>
</table>
</div>

---

## ✨ Features

- **Smart GPU Detection** - Automatically selects the best model for your VRAM
- **7 LLM Tiers** - From 0.5B (CPU) to 14B parameters (24GB+ VRAM)
- **14 Specialized Roles** - Optimized prompts for each target AI
- **Dark Mode UI** - Modern interface with streaming generation
- **Model Cleanup** - Delete downloaded models to free disk space
- **Zero Config** - Works out of the box with Docker
- **Fully Offline** - No API keys or internet required after setup

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
# GPU (NVIDIA) - auto-detects VRAM
docker compose --profile gpu up -d

# CPU only
docker compose --profile cpu up -d
```

Open **http://localhost:7610**

> Models auto-download on first use and persist in `./models/`

### Manual Installation

```bash
# GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install gradio huggingface_hub
python app.py

# CPU only
pip install llama-cpp-python gradio huggingface_hub
python app.py
```

---

## 🎯 Supported Targets

<table>
<tr>
<td width="33%">

### 🎬 Video
- **Wan2.1** - Cinematic text-to-video
- **Wan2.2** - Enhanced motion & physics
- **Hunyuan Video** - Tencent T2V
- **Hunyuan 1.5** - Extended duration + I2V

</td>
<td width="33%">

### 🖼️ Image
- **Stable Diffusion** - SD/SDXL
- **Midjourney** - Artistic style
- **FLUX** - Black Forest Labs
- **DALL-E 3** - OpenAI
- **ComfyUI** - Workflow prompts

</td>
<td width="33%">

### ✍️ Creative
- **Story Writer** - Narratives
- **Code Generator** - Programming
- **Technical Writer** - Docs
- **Marketing Copy** - Ads & CTAs
- **SEO Content** - Blog posts

</td>
</tr>
</table>

---

## 🧠 LLM Options

PromptMill automatically selects the best model based on your GPU:

| VRAM | Model | Size | Quality |
|:-----|:------|:-----|:--------|
| CPU | Qwen2.5 0.5B Q8 | ~1GB | ⭐ |
| 4GB | Qwen2.5 1.5B Q4 | ~2GB | ⭐⭐ |
| 6GB | Qwen2.5 3B Q4 | ~4GB | ⭐⭐⭐ |
| 8GB | Dolphin Mistral 7B Q4 | ~6GB | ⭐⭐⭐⭐ |
| 12GB | Qwen2.5 7B Q6 | ~10GB | ⭐⭐⭐⭐ |
| 16GB+ | Qwen2.5 14B Q4 | ~12GB | ⭐⭐⭐⭐⭐ |
| 24GB+ | Qwen2.5 14B Q8 | ~18GB | ⭐⭐⭐⭐⭐ |

---

## ⚙️ Configuration

The app auto-configures based on your hardware:

- **GPU detected** → Uses all layers on GPU, selects model by VRAM
- **No GPU** → CPU mode with lightweight 0.5B model

Manual override available in the UI for GPU layers and model selection.

---

## 📁 Project Structure

```
PromptMill/
├── app.py              # Main application
├── pyproject.toml      # Project config & dependencies
├── assets/
│   └── logo.svg        # Logo
├── Dockerfile.gpu      # CUDA build
├── Dockerfile.cpu      # CPU build
├── docker-compose.yml  # Docker orchestration
└── models/             # Downloaded LLMs (persisted)
```

---

## 🛠️ Development

Requires [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Install dependencies
uv sync

# Run with uv
uv run python app.py

# Lint & format
uv run ruff check --fix
uv run ruff format
```

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs or request features via [Issues](https://github.com/kekzl/PromptMill/issues)
- Submit pull requests

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[⬆ Back to top](#)**

Made with ❤️ for the AI creative community

</div>
