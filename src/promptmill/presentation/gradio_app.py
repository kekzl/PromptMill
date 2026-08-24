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

from promptmill import __version__
from promptmill.application.services.health_service import HealthService
from promptmill.application.services.model_service import ModelService
from promptmill.application.services.prompt_service import PromptService
from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import RoleCategory
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort
from promptmill.domain.value_objects.prompt_request import PromptGenerationRequest
from promptmill.domain.value_objects.prompt_result import PromptGenerationResult
from promptmill.presentation import history as hist
from promptmill.presentation.api import create_api_router
from promptmill.presentation.examples import examples_for
from promptmill.presentation.theme import create_theme

logger = logging.getLogger(__name__)

# Sentinel for the category filter's "no filter" option.
ALL_CATEGORIES = "All"

# Number of example buttons rendered, split across two rows.
EXAMPLE_BUTTON_COUNT = 6

# Custom CSS for improved dropdown contrast.
# Uses stable, role-based selectors instead of Gradio's build-specific hashed
# Svelte class names (e.g. ``.svelte-xxxxxx``), which change between releases.
CUSTOM_CSS = """
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
"""


@dataclass
class GradioApp:
    """Gradio application for PromptMill.

    This class encapsulates the Gradio UI and coordinates between
    the presentation layer and the application services.
    """

    prompt_service: PromptService
    model_service: ModelService
    health_service: HealthService
    role_repository: RoleRepositoryPort
    assets_dir: Path
    gpu_info: GPUInfo | None
    default_model: Model

    _app: gr.Blocks | None = None

    # =========================================================================
    # UI construction
    # =========================================================================

    def create(self) -> gr.Blocks:
        """Create and configure the Gradio Blocks application.

        Returns:
            Configured Gradio Blocks instance.
        """
        category_choices = [
            ALL_CATEGORIES,
            *(c.value for c in self.role_repository.get_categories()),
        ]
        role_choices = self.role_repository.get_display_names()
        model_choices = self.model_service.get_model_names()

        initial_role = role_choices[0] if role_choices else ""
        initial_examples = self._examples_for_role(initial_role)

        gpu_status = self._gpu_status_text()

        # In Gradio 6 the ``theme`` and ``css`` parameters moved off the Blocks
        # constructor and are supplied at mount/launch time instead.
        with gr.Blocks(title="PromptMill") as app:
            # Session state: examples for the active target, and the prompt
            # history. Both are per-browser-session, never server-global.
            examples_state = gr.State([text for _, text in initial_examples])
            history_state: gr.State = gr.State([])

            gr.HTML(self._create_header_html(gpu_status))

            with gr.Row():
                # Left column - main interaction
                with gr.Column(scale=2):
                    category_filter = gr.Radio(
                        label="Category",
                        choices=category_choices,
                        value=ALL_CATEGORIES,
                        info=f"{self.role_repository.count()} targets available",
                    )

                    role_dropdown = gr.Dropdown(
                        label="Target AI Model",
                        choices=role_choices,
                        value=initial_role or None,
                        info="Type to search, or narrow the list with the category filter",
                    )

                    role_info = gr.Markdown(value=self._get_role_info(initial_role))

                    user_idea = gr.Textbox(
                        label="Your Idea / Request",
                        placeholder="Describe what you want to create, or click an example below...",
                        lines=5,
                        max_lines=10,
                    )

                    # Example buttons; labels follow the selected category.
                    gr.Markdown("**Quick Examples:**")
                    example_buttons: list[gr.Button] = []
                    with gr.Row():
                        example_buttons += [
                            gr.Button(initial_examples[i][0], size="sm") for i in range(3)
                        ]
                    with gr.Row():
                        example_buttons += [
                            gr.Button(initial_examples[i][0], size="sm")
                            for i in range(3, EXAMPLE_BUTTON_COUNT)
                        ]

                    generate_btn = gr.Button("Generate Prompt", variant="primary", size="lg")

                    output = gr.Textbox(
                        label="Generated Prompt",
                        lines=10,
                        max_lines=20,
                        buttons=["copy"],
                        info="Copy this prompt to use with your AI model",
                    )

                    output_stats = gr.Markdown(value="")

                    with gr.Accordion("History (this session)", open=False):
                        history_dropdown = gr.Dropdown(
                            label="Previous generations",
                            choices=[],
                            interactive=True,
                        )
                        with gr.Row():
                            restore_btn = gr.Button("Restore", size="sm")
                            clear_history_btn = gr.Button("Clear", size="sm", variant="stop")

                # Right column - settings
                with gr.Column(scale=1):
                    gr.Markdown(f"### LLM for Prompt Generation\n*{gpu_status}*")

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
                        info="-1 = all layers on GPU, 0 = CPU only. Changing this reloads the model.",
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

            gr.HTML(self._create_footer_html())

            self._setup_event_handlers(
                category_filter=category_filter,
                role_dropdown=role_dropdown,
                role_info=role_info,
                model_dropdown=model_dropdown,
                model_info=model_info,
                user_idea=user_idea,
                generate_btn=generate_btn,
                output=output,
                output_stats=output_stats,
                temperature=temperature,
                max_tokens=max_tokens,
                n_gpu_layers=n_gpu_layers,
                example_buttons=example_buttons,
                examples_state=examples_state,
                history_state=history_state,
                history_dropdown=history_dropdown,
                restore_btn=restore_btn,
                clear_history_btn=clear_history_btn,
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
        category_filter: gr.Radio,
        role_dropdown: gr.Dropdown,
        role_info: gr.Markdown,
        model_dropdown: gr.Dropdown,
        model_info: gr.Markdown,
        user_idea: gr.Textbox,
        generate_btn: gr.Button,
        output: gr.Textbox,
        output_stats: gr.Markdown,
        temperature: gr.Slider,
        max_tokens: gr.Slider,
        n_gpu_layers: gr.Slider,
        example_buttons: list[gr.Button],
        examples_state: gr.State,
        history_state: gr.State,
        history_dropdown: gr.Dropdown,
        restore_btn: gr.Button,
        clear_history_btn: gr.Button,
        refresh_models_btn: gr.Button,
        delete_all_btn: gr.Button,
        model_to_delete: gr.Dropdown,
        delete_one_btn: gr.Button,
        cleanup_result: gr.Markdown,
        models_status: gr.Markdown,
    ) -> None:
        """Set up all event handlers for the UI."""

        # Category filter narrows the target list and re-seeds the examples.
        category_filter.change(
            fn=self._on_category_change,
            inputs=[category_filter],
            outputs=[role_dropdown, role_info, examples_state, *example_buttons],
        )

        # Target change refreshes the description and the example set.
        role_dropdown.change(
            fn=self._on_role_change,
            inputs=[role_dropdown],
            outputs=[role_info, examples_state, *example_buttons],
        )

        # Model change refreshes the description and the GPU layer default.
        model_dropdown.change(
            fn=self._on_model_change,
            inputs=[model_dropdown],
            outputs=[model_info, n_gpu_layers],
        )

        # Example buttons read their text from session state, so the same
        # button serves whatever category is currently selected.
        for index, button in enumerate(example_buttons):
            button.click(
                fn=lambda texts, i=index: self._example_text(texts, i),
                inputs=[examples_state],
                outputs=user_idea,
            )

        generation_inputs = [
            user_idea,
            role_dropdown,
            model_dropdown,
            temperature,
            max_tokens,
            n_gpu_layers,
        ]

        for trigger in (generate_btn.click, user_idea.submit):
            trigger(
                fn=self._generate_prompt,
                inputs=generation_inputs,
                outputs=output,
            ).then(
                fn=self._record_generation,
                inputs=[history_state, user_idea, role_dropdown, model_dropdown, output],
                outputs=[history_state, history_dropdown, output_stats],
            )

        restore_btn.click(
            fn=self._restore_history,
            inputs=[history_state, history_dropdown],
            outputs=[user_idea, role_dropdown, output, output_stats],
        )

        clear_history_btn.click(
            fn=self._clear_history,
            outputs=[history_state, history_dropdown],
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

    # =========================================================================
    # Target selection
    # =========================================================================

    def _role_choices_for(self, category_label: str) -> list[str]:
        """List target display names for a category label.

        Args:
            category_label: A category value or ``ALL_CATEGORIES``.

        Returns:
            Display names of matching targets.
        """
        if category_label == ALL_CATEGORIES:
            return self.role_repository.get_display_names()

        category = RoleCategory.from_string(category_label)
        return [role.display_name for role in self.role_repository.get_by_category(category)]

    def _examples_for_role(self, role_display_name: str) -> tuple[tuple[str, str], ...]:
        """Get the example set matching a target's category.

        Args:
            role_display_name: Target display name.

        Returns:
            Six (label, prompt) pairs.
        """
        role = self.role_repository.get_by_display_name(role_display_name)
        category = role.category if role else RoleCategory.CREATIVE
        return examples_for(category)

    def _example_updates(self, role_display_name: str) -> tuple[Any, ...]:
        """Build the state and button updates for a target's examples.

        Args:
            role_display_name: Target display name.

        Returns:
            Tuple of (example texts, six button updates).
        """
        examples = self._examples_for_role(role_display_name)
        texts = [text for _, text in examples]
        buttons = tuple(gr.update(value=label) for label, _ in examples)
        return (texts, *buttons)

    def _on_category_change(self, category_label: str) -> tuple[Any, ...]:
        """Filter the target list and re-seed the examples.

        Args:
            category_label: Selected category label.

        Returns:
            Tuple of (dropdown update, role info, example texts, button updates).
        """
        choices = self._role_choices_for(category_label)
        selected = choices[0] if choices else ""
        return (
            gr.update(choices=choices, value=selected or None),
            self._get_role_info(selected),
            *self._example_updates(selected),
        )

    def _on_role_change(self, role_display_name: str) -> tuple[Any, ...]:
        """Refresh the description and examples for the selected target.

        Args:
            role_display_name: Selected target display name.

        Returns:
            Tuple of (role info, example texts, button updates).
        """
        return (self._get_role_info(role_display_name), *self._example_updates(role_display_name))

    def _on_model_change(self, model_name: str) -> tuple[str, Any]:
        """Refresh model description and reset the GPU layer slider.

        Args:
            model_name: Selected model display name.

        Returns:
            Tuple of (model info markdown, GPU layer slider update).
        """
        model = self.model_service.get_model_by_name(model_name)
        if model is None:
            return (f"Model not found: {model_name}", gr.update())
        return (self._get_model_info(model_name), gr.update(value=model.n_gpu_layers))

    @staticmethod
    def _example_text(texts: list[str], index: int) -> str:
        """Read one example text out of session state.

        Args:
            texts: Example texts for the active category.
            index: Button index.

        Returns:
            The example text, or empty string if state is missing.
        """
        if not texts or index >= len(texts):
            return ""
        return texts[index]

    # =========================================================================
    # Generation
    # =========================================================================

    def _generate_prompt(
        self,
        user_input: str,
        role_choice: str,
        model_choice: str,
        temperature: float,
        max_tokens: int,
        n_gpu_layers: int,
    ) -> Iterator[str]:
        """Generate a prompt using the selected model and role.

        Args:
            user_input: User's idea/request.
            role_choice: Selected role display name.
            model_choice: Selected model display name.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
            n_gpu_layers: GPU offload override for this generation.

        Yields:
            Generated text chunks.
        """
        if not user_input or not user_input.strip():
            yield "Please enter an idea or description to generate a prompt."
            return

        try:
            request = PromptGenerationRequest(
                user_input=user_input,
                role_display_name=role_choice,
                temperature=temperature,
                max_tokens=int(max_tokens),
            )

            model = self.model_service.get_model_by_name(model_choice)
            if model is None:
                yield f"Model not found: {model_choice}"
                return

            accumulated = ""
            for chunk in self.prompt_service.generate(
                request, model, n_gpu_layers_override=int(n_gpu_layers)
            ):
                accumulated += chunk
                yield accumulated

        except Exception as e:
            logger.exception("Generation error")
            yield f"Error: {e}"

    def _record_generation(
        self,
        history: list[hist.HistoryEntry],
        user_input: str,
        role_choice: str,
        model_choice: str,
        generated: str,
    ) -> tuple[list[hist.HistoryEntry], Any, str]:
        """Append a finished generation to the session history.

        Args:
            history: Existing history, newest first.
            user_input: The idea that produced the prompt.
            role_choice: Target display name.
            model_choice: Model display name.
            generated: The generated prompt text.

        Returns:
            Tuple of (history, history dropdown update, stats markdown).
        """
        if not generated.strip() or not user_input.strip():
            return (history, gr.update(), "")

        result = PromptGenerationResult(
            content=generated,
            model_used=model_choice,
            role_used=role_choice,
        )
        entry = hist.HistoryEntry(
            user_input=user_input,
            role_display_name=role_choice,
            model_name=model_choice,
            result=result,
        )
        updated = hist.add_entry(history, entry)
        return (
            updated,
            gr.update(choices=hist.labels(updated), value=None),
            self._format_stats(result),
        )

    def _restore_history(
        self,
        history: list[hist.HistoryEntry],
        label: str,
    ) -> tuple[Any, Any, Any, str]:
        """Restore inputs and output from a history entry.

        Args:
            history: Session history, newest first.
            label: Selected history label.

        Returns:
            Tuple of (user idea, role dropdown, output, stats markdown).
        """
        entry = hist.find_by_label(history, label) if label else None
        if entry is None:
            return (gr.update(), gr.update(), gr.update(), "")

        return (
            entry.user_input,
            gr.update(value=entry.role_display_name),
            entry.result.content,
            self._format_stats(entry.result),
        )

    @staticmethod
    def _clear_history() -> tuple[list[hist.HistoryEntry], Any]:
        """Drop all history for this session.

        Returns:
            Tuple of (empty history, cleared dropdown update).
        """
        return ([], gr.update(choices=[], value=None))

    @staticmethod
    def _format_stats(result: PromptGenerationResult) -> str:
        """Render the character and word count line.

        Args:
            result: The generation result.

        Returns:
            Markdown stats line.
        """
        return f"*{result.char_count} characters · {result.word_count} words · {result.model_used}*"

    # =========================================================================
    # Info panels
    # =========================================================================

    def _get_role_info(self, role_choice: str) -> str:
        """Get role description for display."""
        role = self.role_repository.get_by_display_name(role_choice)
        if role:
            return f"**{role.description}**"
        return ""

    def _get_model_info(self, model_choice: str) -> str:
        """Get model description for display."""
        model = self.model_service.get_model_by_name(model_choice)
        if model:
            return (
                f"**{model.description}**\n\n"
                f"VRAM usage: {model.vram_required} · Context: {model.context_length:,} tokens"
            )
        return ""

    def _gpu_status_text(self) -> str:
        """Build the GPU status line shown in header and sidebar."""
        if self.gpu_info and self.gpu_info.is_available:
            return f"{self.gpu_info.name} ({self.gpu_info.vram_gb:.0f}GB VRAM)"
        return "CPU mode (no GPU detected)"

    # =========================================================================
    # Model management
    # =========================================================================

    def _refresh_models_list(self) -> tuple[str, Any, Any, Any]:
        """Refresh the list of downloaded models."""
        models = self.model_service.get_available_models()
        downloaded = [m for m in models if self.model_service.is_model_downloaded(m)]

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

    # =========================================================================
    # Chrome
    # =========================================================================

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
                <a href="/docs" style="color: #818cf8; text-decoration: none;">API</a> |
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

    # =========================================================================
    # Server
    # =========================================================================

    def create_fastapi_app(self) -> FastAPI:
        """Create the FastAPI app with health, REST API and Gradio mounted.

        Returns:
            Configured FastAPI application.
        """
        if self._app is None:
            self.create()

        fastapi_app = FastAPI(
            title="PromptMill",
            version=__version__,
            description="Local prompt generator for image, video, audio, 3D and writing targets.",
        )

        fastapi_app.include_router(
            create_api_router(
                prompt_service=self.prompt_service,
                model_service=self.model_service,
                health_service=self.health_service,
                role_repository=self.role_repository,
            )
        )

        # Mount Gradio app at root. Theme and CSS are supplied here because
        # Gradio 6 removed them from the Blocks constructor.
        return gr.mount_gradio_app(
            fastapi_app,
            self._app,
            path="/",
            theme=create_theme(),
            css=CUSTOM_CSS,
        )

    def launch(self, host: str, port: int) -> None:
        """Launch the Gradio application.

        Args:
            host: Server host address.
            port: Server port number.
        """
        fastapi_app = self.create_fastapi_app()

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
