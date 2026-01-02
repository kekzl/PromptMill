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
        # High contrast text - brighter white
        body_text_color="#fafafa",
        body_text_color_dark="#fafafa",
        body_text_color_subdued="#d4d4d8",
        body_text_color_subdued_dark="#d4d4d8",
        # Block styling
        block_background_fill="#18181b",
        block_background_fill_dark="#18181b",
        block_border_color="#3f3f46",
        block_border_color_dark="#3f3f46",
        block_label_background_fill="#18181b",
        block_label_background_fill_dark="#18181b",
        block_label_text_color="*primary_300",
        block_label_text_color_dark="*primary_300",
        block_title_text_color="#ffffff",
        block_title_text_color_dark="#ffffff",
        # Input fields - high contrast
        input_background_fill="#27272a",
        input_background_fill_dark="#27272a",
        input_border_color="#52525b",
        input_border_color_dark="#52525b",
        input_border_color_focus="*primary_400",
        input_border_color_focus_dark="*primary_400",
        input_placeholder_color="#a1a1aa",
        input_placeholder_color_dark="#a1a1aa",
        # Primary button - vibrant
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_dark="*primary_500",
        button_primary_background_fill_hover="*primary_400",
        button_primary_background_fill_hover_dark="*primary_400",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        # Secondary buttons - clearer
        button_secondary_background_fill="#3f3f46",
        button_secondary_background_fill_dark="#3f3f46",
        button_secondary_background_fill_hover="#52525b",
        button_secondary_background_fill_hover_dark="#52525b",
        button_secondary_text_color="#fafafa",
        button_secondary_text_color_dark="#fafafa",
        # Shadows
        shadow_drop="0 4px 6px -1px rgba(0, 0, 0, 0.4)",
        shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.4)",
    )
