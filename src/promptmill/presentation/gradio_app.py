"""Gradio UI application for PromptMill."""

import base64
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from promptmill import __version__
from promptmill.application.services.health_service import HealthService
from promptmill.application.services.model_service import ModelService
from promptmill.application.services.prompt_service import PromptService
from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.value_objects.prompt_request import PromptGenerationRequest
from promptmill.presentation.theme import create_theme

logger = logging.getLogger(__name__)

# Custom CSS for improved dropdown contrast
CUSTOM_CSS = """
/* Dropdown menu styling - dark background with high contrast text */
.svelte-1gfkn6j {
    background-color: #18181b !important;
    border-color: #3f3f46 !important;
}

/* Dropdown options */
.svelte-1gfkn6j ul {
    background-color: #18181b !important;
}

.svelte-1gfkn6j li {
    color: #fafafa !important;
    background-color: #18181b !important;
}

.svelte-1gfkn6j li:hover {
    background-color: #27272a !important;
    color: #ffffff !important;
}

.svelte-1gfkn6j li[aria-selected="true"] {
    background-color: #3f3f46 !important;
    color: #ffffff !important;
}

/* Generic dropdown/listbox styling */
[role="listbox"] {
    background-color: #18181b !important;
    border-color: #3f3f46 !important;
}

[role="option"] {
    color: #fafafa !important;
    background-color: #18181b !important;
}

[role="option"]:hover {
    background-color: #27272a !important;
}

[role="option"][aria-selected="true"] {
    background-color: #3f3f46 !important;
}

/* Input text color */
input, textarea, select {
    color: #fafafa !important;
}

/* Ensure dropdown text is visible */
.wrap.svelte-1gfkn6j {
    color: #fafafa !important;
}
"""

# Example prompts for quick start
EXAMPLE_PROMPTS = [
    (
        "Samurai in Cherry Blossoms",
        "A lone samurai walking slowly through a path of falling cherry blossoms at golden hour sunset, katana at his side, petals swirling in the gentle breeze",
    ),
    (
        "Timelapse Flower Bloom",
        "Macro timelapse of a delicate flower bud slowly opening and blooming in a sunlit garden, dewdrops glistening on petals, soft bokeh background",
    ),
    (
        "Ocean Waves Aerial",
        "Cinematic aerial drone shot of powerful turquoise ocean waves crashing against dramatic rocky cliffs, white foam spray, golden hour lighting",
    ),
    (
        "Cyberpunk Portrait",
        "Close-up portrait of a cyberpunk warrior with glowing neon tattoos, rain-soaked face, reflections of holographic billboards, moody night scene",
    ),
    (
        "Cozy Cabin Snow",
        "Cozy wooden cabin nestled in snowy mountains at twilight, warm light glowing from windows, smoke rising from chimney, fresh snowfall",
    ),
    (
        "Astronaut on Mars",
        "An astronaut in a detailed spacesuit walking across the rusty red Martian surface, Earth visible in the distant sky, dramatic shadows",
    ),
]


@dataclass
class GradioApp:
    """Gradio application for PromptMill.

    This class encapsulates the Gradio UI and coordinates between
    the presentation layer and the application services.
    """

    prompt_service: PromptService
    model_service: ModelService
    health_service: HealthService
    assets_dir: Path
    gpu_info: GPUInfo | None
    default_model: Model

    _app: gr.Blocks | None = None

    def create(self) -> gr.Blocks:
        """Create and configure the Gradio Blocks application.

        Returns:
            Configured Gradio Blocks instance.
        """
        role_choices = (
            self.prompt_service.generate_prompt_use_case.role_repository.get_display_names()
        )
        model_choices = self.model_service.get_model_names()

        # Build GPU status string
        if self.gpu_info and self.gpu_info.is_available:
            gpu_status = f"{self.gpu_info.name} ({self.gpu_info.vram_gb:.0f}GB VRAM)"
        else:
            gpu_status = "CPU mode (no GPU detected)"

        with gr.Blocks(title="PromptMill", theme=create_theme(), css=CUSTOM_CSS) as app:
            # Header
            gr.HTML(self._create_header_html(gpu_status))

            with gr.Row():
                # Left column - main interaction
                with gr.Column(scale=2):
                    role_dropdown = gr.Dropdown(
                        label="Target AI Model",
                        choices=role_choices,
                        value=role_choices[0] if role_choices else None,
                        info="Select the AI model you're generating prompts for",
                    )

                    role_info = gr.Markdown(
                        value=self._get_role_info(role_choices[0]) if role_choices else ""
                    )

                    user_idea = gr.Textbox(
                        label="Your Idea / Request",
                        placeholder="Describe what you want to create, or click an example below...",
                        lines=5,
                        max_lines=10,
                    )

                    # Example buttons
                    gr.Markdown("**Quick Examples:**")
                    with gr.Row():
                        ex_btns_row1 = [
                            gr.Button(EXAMPLE_PROMPTS[i][0], size="sm") for i in range(3)
                        ]
                    with gr.Row():
                        ex_btns_row2 = [
                            gr.Button(EXAMPLE_PROMPTS[i][0], size="sm") for i in range(3, 6)
                        ]

                    generate_btn = gr.Button("Generate Prompt", variant="primary", size="lg")

                    output = gr.Textbox(
                        label="Generated Prompt",
                        lines=10,
                        max_lines=20,
                        show_copy_button=True,
                        info="Copy this prompt to use with your AI model",
                    )

                # Right column - settings
                with gr.Column(scale=1):
                    if self.gpu_info and self.gpu_info.is_available:
                        gr.Markdown(
                            f"### LLM for Prompt Generation\n*Auto-detected: {self.gpu_info.vram_gb:.0f}GB VRAM*"
                        )
                    else:
                        gr.Markdown("### LLM for Prompt Generation\n*No GPU detected - using CPU*")

                    model_dropdown = gr.Dropdown(
                        label="Select by Your GPU VRAM",
                        choices=model_choices,
                        value=self.default_model.name,
                        info="Auto-selected based on detected VRAM"
                        if self.gpu_info and self.gpu_info.is_available
                        else "Select manually or use CPU model",
                    )

                    model_info = gr.Markdown(value=self._get_model_info(self.default_model.name))

                    gr.Markdown("### Output Settings")

                    temperature = gr.Slider(
                        label="Creativity (Temperature)",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        info="0.3-0.5 precise, 0.7-1.0 creative, 1.0+ experimental",
                    )

                    max_tokens = gr.Slider(
                        label="Max Length (Tokens)",
                        minimum=100,
                        maximum=2000,
                        value=256,
                        step=50,
                        info="Video prompts: 150-300, Image prompts: 75-150",
                    )

                    gr.Markdown("### Advanced")

                    n_gpu_layers = gr.Slider(
                        label="GPU Layers",
                        minimum=-1,
                        maximum=100,
                        value=self.default_model.n_gpu_layers,
                        step=1,
                        info="-1 = all layers on GPU, 0 = CPU only",
                    )

                    # Model Management
                    with gr.Accordion("Model Management", open=False):
                        models_status = gr.Markdown(value="Click refresh to see downloaded models")
                        with gr.Row():
                            refresh_models_btn = gr.Button("Refresh", size="sm")
                            delete_all_btn = gr.Button("Delete All", size="sm", variant="stop")
                        model_to_delete = gr.Dropdown(
                            label="Select Model to Delete",
                            choices=[],
                            interactive=True,
                            visible=False,
                        )
                        delete_one_btn = gr.Button(
                            "Delete Selected", size="sm", variant="stop", visible=False
                        )
                        cleanup_result = gr.Markdown(visible=False)

                    gr.HTML(
                        """
                        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #475569;">
                            <p style="color: #64748b; font-size: 12px; margin: 0;">
                                Models auto-download on first use<br>
                                Changing model will free current memory
                            </p>
                        </div>
                        """
                    )

            # Footer
            gr.HTML(self._create_footer_html())

            # Event handlers
            self._setup_event_handlers(
                app,
                role_dropdown=role_dropdown,
                role_info=role_info,
                model_dropdown=model_dropdown,
                model_info=model_info,
                user_idea=user_idea,
                generate_btn=generate_btn,
                output=output,
                temperature=temperature,
                max_tokens=max_tokens,
                n_gpu_layers=n_gpu_layers,
                ex_btns_row1=ex_btns_row1,
                ex_btns_row2=ex_btns_row2,
                refresh_models_btn=refresh_models_btn,
                delete_all_btn=delete_all_btn,
                model_to_delete=model_to_delete,
                delete_one_btn=delete_one_btn,
                cleanup_result=cleanup_result,
                models_status=models_status,
            )

        self._app = app
        return app

    def _setup_event_handlers(
        self,
        _app: gr.Blocks,
        role_dropdown: gr.Dropdown,
        role_info: gr.Markdown,
        model_dropdown: gr.Dropdown,
        model_info: gr.Markdown,
        user_idea: gr.Textbox,
        generate_btn: gr.Button,
        output: gr.Textbox,
        temperature: gr.Slider,
        max_tokens: gr.Slider,
        n_gpu_layers: gr.Slider,
        ex_btns_row1: list[gr.Button],
        ex_btns_row2: list[gr.Button],
        refresh_models_btn: gr.Button,
        delete_all_btn: gr.Button,
        model_to_delete: gr.Dropdown,
        delete_one_btn: gr.Button,
        cleanup_result: gr.Markdown,
        models_status: gr.Markdown,
    ) -> None:
        """Set up all event handlers for the UI."""

        # Role info update
        role_dropdown.change(
            fn=self._get_role_info,
            inputs=[role_dropdown],
            outputs=[role_info],
        )

        # Model info update
        model_dropdown.change(
            fn=self._get_model_info,
            inputs=[model_dropdown],
            outputs=[model_info],
        )

        # Example buttons
        for i, btn in enumerate(ex_btns_row1):
            btn.click(fn=lambda p=EXAMPLE_PROMPTS[i][1]: p, outputs=user_idea)
        for i, btn in enumerate(ex_btns_row2):
            btn.click(fn=lambda p=EXAMPLE_PROMPTS[i + 3][1]: p, outputs=user_idea)

        # Generate button
        generate_btn.click(
            fn=self._generate_prompt,
            inputs=[
                user_idea,
                role_dropdown,
                model_dropdown,
                temperature,
                max_tokens,
                n_gpu_layers,
            ],
            outputs=output,
        )

        # Submit on enter
        user_idea.submit(
            fn=self._generate_prompt,
            inputs=[
                user_idea,
                role_dropdown,
                model_dropdown,
                temperature,
                max_tokens,
                n_gpu_layers,
            ],
            outputs=output,
        )

        # Model management
        refresh_models_btn.click(
            fn=self._refresh_models_list,
            outputs=[models_status, model_to_delete, delete_one_btn, cleanup_result],
        )

        delete_one_btn.click(
            fn=self._delete_one_model,
            inputs=[model_to_delete],
            outputs=[cleanup_result],
        ).then(
            fn=self._refresh_models_list,
            outputs=[models_status, model_to_delete, delete_one_btn, cleanup_result],
        )

        delete_all_btn.click(
            fn=self._delete_all_models,
            outputs=[cleanup_result],
        ).then(
            fn=self._refresh_models_list,
            outputs=[models_status, model_to_delete, delete_one_btn, cleanup_result],
        )

    def _generate_prompt(
        self,
        user_input: str,
        role_choice: str,
        model_choice: str,
        temperature: float,
        max_tokens: int,
        _n_gpu_layers: int,
    ) -> Iterator[str]:
        """Generate a prompt using the selected model and role.

        Args:
            user_input: User's idea/request.
            role_choice: Selected role display name.
            model_choice: Selected model display name.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
            _n_gpu_layers: GPU layers override (currently unused in generation).

        Yields:
            Generated text chunks.
        """
        if not user_input or not user_input.strip():
            yield "Please enter an idea or description to generate a prompt."
            return

        try:
            # Create request
            request = PromptGenerationRequest(
                user_input=user_input,
                role_display_name=role_choice,
                temperature=temperature,
                max_tokens=int(max_tokens),
            )

            # Get model
            model = self.model_service.get_model_by_name(model_choice)
            if model is None:
                yield f"Model not found: {model_choice}"
                return

            # Generate with streaming
            accumulated = ""
            for chunk in self.prompt_service.generate(request, model):
                accumulated += chunk
                yield accumulated

        except Exception as e:
            logger.exception("Generation error")
            yield f"Error: {e}"

    def _get_role_info(self, role_choice: str) -> str:
        """Get role description for display."""
        role_repo = self.prompt_service.generate_prompt_use_case.role_repository
        role = role_repo.get_by_display_name(role_choice)
        if role:
            return f"**{role.description}**"
        return ""

    def _get_model_info(self, model_choice: str) -> str:
        """Get model description for display."""
        model = self.model_service.get_model_by_name(model_choice)
        if model:
            return f"**{model.description}**\n\nVRAM usage: {model.vram_required}"
        return ""

    def _refresh_models_list(self) -> tuple[str, Any, Any, Any]:
        """Refresh the list of downloaded models."""
        models = self.model_service.get_available_models()
        downloaded = []

        for model in models:
            if self.model_service.is_model_downloaded(model):
                downloaded.append(model)

        if not downloaded:
            return (
                "**No models downloaded yet**\n\nModels will be downloaded on first use.",
                gr.update(choices=[], visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        total_usage = self.model_service.get_disk_usage_formatted()
        lines = [f"**Downloaded Models** ({len(downloaded)} models, {total_usage} total)\n"]
        choices = []

        for model in downloaded:
            lines.append(f"- **{model.description}**")
            choices.append(model.name)

        return (
            "\n".join(lines),
            gr.update(choices=choices, visible=True, value=None),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def _delete_one_model(self, model_name: str) -> Any:
        """Delete a single model."""
        if not model_name:
            return gr.update(value="Please select a model to delete", visible=True)

        model = self.model_service.get_model_by_name(model_name)
        if model is None:
            return gr.update(value=f"Model not found: {model_name}", visible=True)

        success = self.model_service.delete_model(model)
        if success:
            return gr.update(value=f"Deleted: {model.description}", visible=True)
        return gr.update(value=f"Failed to delete: {model_name}", visible=True)

    def _delete_all_models(self) -> Any:
        """Delete all downloaded models."""
        models = self.model_service.get_available_models()
        count = 0

        for model in models:
            if self.model_service.is_model_downloaded(model) and self.model_service.delete_model(
                model
            ):
                count += 1

        if count > 0:
            return gr.update(value=f"Deleted {count} models", visible=True)
        return gr.update(value="No models to delete", visible=True)

    def _create_header_html(self, gpu_status: str) -> str:
        """Create header HTML with logo and status."""
        logo_html = self._get_logo_html()
        return f"""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            {logo_html}
            <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 14px;">
                AI-powered prompt generator for video, image, and creative content
            </p>
            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 12px;">
                {gpu_status}
            </p>
        </div>
        """

    def _create_footer_html(self) -> str:
        """Create footer HTML."""
        return f"""
        <div style="text-align: center; padding: 20px 0; margin-top: 20px; border-top: 1px solid #334155;">
            <p style="color: #64748b; font-size: 12px; margin: 0;">
                PromptMill v{__version__} |
                <a href="https://github.com/kekzl/PromptMill" style="color: #818cf8; text-decoration: none;">GitHub</a>
            </p>
        </div>
        """

    def _get_logo_html(self) -> str:
        """Load and return the logo as base64 HTML."""
        logo_path = self.assets_dir / "logo.svg"
        try:
            if logo_path.exists():
                content = logo_path.read_text()
                encoded = base64.b64encode(content.encode()).decode()
                return f'<img src="data:image/svg+xml;base64,{encoded}" alt="PromptMill" style="height: 48px; margin-bottom: 8px;">'
        except Exception as e:
            logger.warning(f"Failed to load logo: {e}")
        return '<h1 style="color: #818cf8; margin: 0;">PromptMill</h1>'

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with health endpoint and Gradio mounted.

        Returns:
            Configured FastAPI application.
        """
        fastapi_app = FastAPI(title="PromptMill", version=__version__)

        @fastapi_app.get("/health")
        def health_check() -> JSONResponse:
            """Health check endpoint for container orchestration."""
            status = self.health_service.get_status()
            return JSONResponse(content=dict(status))

        # Mount Gradio app at root
        fastapi_app = gr.mount_gradio_app(fastapi_app, self._app, path="/")

        return fastapi_app

    def launch(self, host: str, port: int) -> None:
        """Launch the Gradio application.

        Args:
            host: Server host address.
            port: Server port number.
        """
        if self._app is None:
            self.create()

        fastapi_app = self._create_fastapi_app()

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
