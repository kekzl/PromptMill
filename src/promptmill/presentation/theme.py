"""Custom Gradio theme for PromptMill."""

import gradio as gr


def create_theme() -> gr.themes.Base:
    """Create a custom dark theme for PromptMill.

    Returns:
        Gradio theme instance with custom dark styling.
    """
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_200",
        body_text_color_dark="*neutral_200",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_color="*neutral_800",
        block_border_color_dark="*neutral_800",
        block_label_background_fill="*neutral_900",
        block_label_background_fill_dark="*neutral_900",
        input_background_fill="*neutral_800",
        input_background_fill_dark="*neutral_800",
        input_border_color="*neutral_700",
        input_border_color_dark="*neutral_700",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_dark="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_background_fill_hover_dark="*primary_500",
        button_secondary_background_fill="*neutral_700",
        button_secondary_background_fill_dark="*neutral_700",
    )
