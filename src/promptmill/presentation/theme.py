"""Custom Gradio theme for PromptMill."""

import gradio as gr


def create_theme() -> gr.themes.Base:
    """Create a custom dark theme for PromptMill.

    Returns:
        Gradio theme instance with custom dark styling.
    """
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        # Background colors - rich dark
        body_background_fill="#0f0f12",
        body_background_fill_dark="#0f0f12",
        # High contrast text
        body_text_color="#e4e4e7",
        body_text_color_dark="#e4e4e7",
        body_text_color_subdued="#a1a1aa",
        body_text_color_subdued_dark="#a1a1aa",
        # Block styling
        block_background_fill="#18181b",
        block_background_fill_dark="#18181b",
        block_border_color="#27272a",
        block_border_color_dark="#27272a",
        block_label_background_fill="#18181b",
        block_label_background_fill_dark="#18181b",
        block_label_text_color="*primary_400",
        block_label_text_color_dark="*primary_400",
        block_title_text_color="#fafafa",
        block_title_text_color_dark="#fafafa",
        # Input fields - clear visibility
        input_background_fill="#27272a",
        input_background_fill_dark="#27272a",
        input_border_color="#3f3f46",
        input_border_color_dark="#3f3f46",
        input_border_color_focus="*primary_500",
        input_border_color_focus_dark="*primary_500",
        input_placeholder_color="#71717a",
        input_placeholder_color_dark="#71717a",
        # Primary button - vibrant
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_dark="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_background_fill_hover_dark="*primary_500",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        # Secondary buttons
        button_secondary_background_fill="#3f3f46",
        button_secondary_background_fill_dark="#3f3f46",
        button_secondary_background_fill_hover="#52525b",
        button_secondary_background_fill_hover_dark="#52525b",
        button_secondary_text_color="#e4e4e7",
        button_secondary_text_color_dark="#e4e4e7",
        # Shadows
        shadow_drop="0 4px 6px -1px rgba(0, 0, 0, 0.3)",
        shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.3)",
    )
