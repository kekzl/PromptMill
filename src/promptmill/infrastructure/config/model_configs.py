"""Model configurations by VRAM tier."""

from promptmill.domain.entities.model import Model

# Model configurations organized by VRAM requirements
MODEL_CONFIGS: dict[str, Model] = {
    "cpu_only": Model(
        key="cpu_only",
        name="CPU Only (2-4GB RAM)",
        repo_id="bartowski/Dolphin3.0-Llama3.2-1B-GGUF",
        filename="Dolphin3.0-Llama3.2-1B-Q8_0.gguf",
        context_length=4096,
        n_gpu_layers=0,  # CPU only
        description="Dolphin 3.0 1B Q8 - Uncensored, lightweight",
        vram_required="~1GB",
        revision="main",
    ),
    "4gb_vram": Model(
        key="4gb_vram",
        name="4GB VRAM (GTX 1650, RTX 3050)",
        repo_id="bartowski/Dolphin3.0-Llama3.2-3B-GGUF",
        filename="Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf",
        context_length=4096,
        n_gpu_layers=-1,  # All layers on GPU
        description="Dolphin 3.0 3B Q4_K_M - Uncensored, good balance",
        vram_required="~2.5GB",
        revision="main",
    ),
    "6gb_vram": Model(
        key="6gb_vram",
        name="6GB VRAM (RTX 2060, RTX 3060)",
        repo_id="bartowski/Dolphin3.0-Llama3.2-3B-GGUF",
        filename="Dolphin3.0-Llama3.2-3B-Q8_0.gguf",
        context_length=4096,
        n_gpu_layers=-1,
        description="Dolphin 3.0 3B Q8 - Uncensored, high quality",
        vram_required="~4GB",
        revision="main",
    ),
    "8gb_vram": Model(
        key="8gb_vram",
        name="8GB VRAM (RTX 3070, RTX 4060)",
        repo_id="bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        filename="Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf",
        context_length=8192,
        n_gpu_layers=-1,
        description="Dolphin 3.0 8B Q4_K_M - Uncensored, excellent",
        vram_required="~6GB",
        revision="main",
    ),
    "12gb_vram": Model(
        key="12gb_vram",
        name="12GB VRAM (RTX 3060 12GB, RTX 4070)",
        repo_id="bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        filename="Dolphin3.0-Llama3.1-8B-Q6_K_L.gguf",
        context_length=8192,
        n_gpu_layers=-1,
        description="Dolphin 3.0 8B Q6_K_L - Uncensored, premium",
        vram_required="~10GB",
        revision="main",
    ),
    "16gb_vram": Model(
        key="16gb_vram",
        name="16GB+ VRAM (RTX 4080, RTX 4090)",
        repo_id="bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        filename="Dolphin3.0-Llama3.1-8B-Q8_0.gguf",
        context_length=8192,
        n_gpu_layers=-1,
        description="Dolphin 3.0 8B Q8 - Uncensored, maximum quality",
        vram_required="~12GB",
        revision="main",
    ),
    "24gb_vram": Model(
        key="24gb_vram",
        name="24GB+ VRAM (RTX 3090, RTX 4090)",
        repo_id="bartowski/dolphin-2.9.4-llama3.1-8b-GGUF",
        filename="dolphin-2.9.4-llama3.1-8b-Q8_0.gguf",
        context_length=131072,  # 128K context
        n_gpu_layers=-1,
        description="Dolphin 2.9.4 8B Q8 - Uncensored, maximum precision",
        vram_required="~10GB",
        revision="main",
    ),
}

# Ordered list of model keys for display
MODEL_KEYS_ORDERED: list[str] = [
    "cpu_only",
    "4gb_vram",
    "6gb_vram",
    "8gb_vram",
    "12gb_vram",
    "16gb_vram",
    "24gb_vram",
]


def get_model_by_key(key: str) -> Model | None:
    """Get a model by its key.

    Args:
        key: Model configuration key.

    Returns:
        Model if found, None otherwise.
    """
    return MODEL_CONFIGS.get(key)


def get_model_by_name(name: str) -> Model | None:
    """Get a model by its display name.

    Args:
        name: Model display name.

    Returns:
        Model if found, None otherwise.
    """
    for model in MODEL_CONFIGS.values():
        if model.name == name:
            return model
    return None


def get_all_models() -> list[Model]:
    """Get all models in order.

    Returns:
        List of all Model configurations.
    """
    return [MODEL_CONFIGS[key] for key in MODEL_KEYS_ORDERED]


def get_model_names() -> list[str]:
    """Get all model display names in order.

    Returns:
        List of model display names.
    """
    return [MODEL_CONFIGS[key].name for key in MODEL_KEYS_ORDERED]


def select_model_by_vram(vram_mb: int) -> Model:
    """Select the optimal model based on available VRAM.

    Args:
        vram_mb: Available VRAM in megabytes.

    Returns:
        Best model for the available VRAM.
    """
    vram_gb = vram_mb / 1024

    match vram_gb:
        case v if v >= 20:
            return MODEL_CONFIGS["24gb_vram"]
        case v if v >= 14:
            return MODEL_CONFIGS["16gb_vram"]
        case v if v >= 10:
            return MODEL_CONFIGS["12gb_vram"]
        case v if v >= 7:
            return MODEL_CONFIGS["8gb_vram"]
        case v if v >= 5:
            return MODEL_CONFIGS["6gb_vram"]
        case v if v >= 3:
            return MODEL_CONFIGS["4gb_vram"]
        case _:
            return MODEL_CONFIGS["cpu_only"]
