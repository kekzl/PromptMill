"""Entry point for PromptMill.

This module provides the main entry point for running PromptMill
as a module: `python -m promptmill`
"""

import atexit
import logging
import signal
from typing import Any

from promptmill import __version__
from promptmill.container import Container
from promptmill.infrastructure.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("promptmill")


def main() -> None:
    """Main entry point for PromptMill."""
    # Load settings
    settings = Settings.from_environment()

    # Create container
    container = Container(settings=settings)

    # Register shutdown handler
    def shutdown_handler() -> None:
        logger.info("Shutting down...")
        container.shutdown()
        logger.info("Shutdown complete")

    atexit.register(shutdown_handler)

    # Handle SIGTERM gracefully (for Docker)
    def signal_handler(signum: int, _frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        shutdown_handler()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    # Log startup info
    logger.info("=" * 50)
    logger.info("PromptMill v%s", __version__)
    logger.info("AI Prompt Generator")
    logger.info("=" * 50)

    gpu_info = container.detected_gpu
    if gpu_info and gpu_info.is_available:
        logger.info("GPU: %s", gpu_info.name)
        logger.info("VRAM: %.1f GB (%d MB)", gpu_info.vram_gb, gpu_info.vram_mb)
    else:
        logger.info("GPU: Not detected (CPU mode)")

    default_model = container.default_model
    logger.info("Auto-selected model: %s", default_model.name)
    logger.info("Available roles: %d", container.role_repository.count())
    logger.info("Starting server on %s:%d", settings.host, settings.port)

    # Create and launch the Gradio app
    app = container.gradio_app
    app.create()
    app.launch(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
