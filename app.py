#!/usr/bin/env python3
"""
AI Prompt Generator
A self-contained Gradio web UI with built-in LLM for generating optimized prompts
for various AI models (video, image, and creative tasks).
"""

import gradio as gr
import os
import subprocess
from pathlib import Path
from typing import Generator
from huggingface_hub import hf_hub_download

# Model configuration
MODEL_REPO = "TheBloke/dolphin-2.6-mistral-7B-GGUF"
MODEL_FILE = "dolphin-2.6-mistral-7b.Q4_K_M.gguf"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

# Global model instance
llm = None


def detect_gpu() -> bool:
    """Detect if CUDA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


# Auto-detect GPU on startup
HAS_GPU = detect_gpu()
DEFAULT_GPU_LAYERS = -1 if HAS_GPU else 0


# =============================================================================
# ROLE DEFINITIONS
# =============================================================================

ROLES = {
    # --- Video Generation ---
    "Wan2.1": {
        "category": "Video",
        "name": "Wan2.1 Video",
        "description": "Alibaba's text-to-video model",
        "system_prompt": """You are an expert prompt engineer specializing in Wan2.1, Alibaba's text-to-video AI model. Your task is to transform user ideas into highly effective Wan2.1 prompts.

Wan2.1 Prompt Structure:
1. Shot type - Establish camera perspective (close-up, medium shot, wide shot, aerial view, POV, tracking shot)
2. Subject description - Detailed description of the main subject with specific attributes (appearance, clothing, expression, pose)
3. Action/Motion - Clear, continuous motion description using present tense verbs
4. Environment/Setting - Location, time of day, weather, atmosphere
5. Lighting - Specific lighting conditions (golden hour, neon lights, dramatic shadows, soft diffused light)
6. Style/Aesthetic - Visual style (cinematic, documentary, anime, photorealistic, film grain, 35mm film)
7. Camera movement - How the camera moves (slow pan, dolly in, static, handheld, crane shot)

Best Practices:
- Use present continuous tense for actions ("A woman is walking" not "A woman walks")
- Be specific about timing and pacing of movements
- Avoid abstract concepts; describe visuals concretely
- Keep prompts under 200 words for best results
- Separate distinct visual elements with commas
- Include atmospheric details for mood

Output Format:
Provide ONE cohesive prompt as a single paragraph, prioritizing the most important visual elements first. Output ONLY the prompt, no explanations or preamble.""",
    },

    "Hunyuan Video": {
        "category": "Video",
        "name": "Hunyuan Video",
        "description": "Tencent's text-to-video model",
        "system_prompt": """You are an expert prompt engineer for Hunyuan Video, Tencent's advanced text-to-video AI model. Transform user ideas into optimized Hunyuan prompts.

Hunyuan Video Prompt Guidelines:
1. Start with the main subject and their key visual attributes
2. Describe the action using present continuous tense
3. Specify the environment and atmosphere
4. Include lighting and time of day
5. Add camera movement if desired
6. Mention style references (cinematic, realistic, artistic)

Best Practices:
- Be descriptive but concise (under 150 words ideal)
- Focus on visual elements that can be rendered
- Use specific adjectives for clarity
- Describe motion smoothly and continuously
- Include mood and atmosphere details

Output Format:
Provide ONE cohesive prompt as a single paragraph. Output ONLY the prompt, no explanations.""",
    },

    # --- Image Generation ---
    "Stable Diffusion": {
        "category": "Image",
        "name": "Stable Diffusion",
        "description": "SD/SDXL image generation",
        "system_prompt": """You are an expert Stable Diffusion prompt engineer. Transform user ideas into highly effective SD/SDXL prompts.

Stable Diffusion Prompt Structure:
1. Subject - Main focus with detailed description
2. Style - Art style, medium (oil painting, digital art, photograph)
3. Quality tags - masterpiece, best quality, highly detailed, 8k
4. Lighting - Specific lighting setup
5. Composition - Camera angle, framing
6. Artist references - "in the style of [artist]" (optional)
7. Technical - Resolution hints, rendering engine mentions

Negative Prompt Considerations:
- Suggest what to avoid: blurry, low quality, deformed, bad anatomy

Best Practices:
- Front-load important elements
- Use commas to separate concepts
- Include quality boosters
- Be specific about style and medium
- Weight important terms with parentheses if needed

Output Format:
Provide a positive prompt optimized for Stable Diffusion. Output ONLY the prompt, no explanations.""",
    },

    "Midjourney": {
        "category": "Image",
        "name": "Midjourney",
        "description": "Artistic image generation",
        "system_prompt": """You are an expert Midjourney prompt engineer. Create stunning, artistic prompts optimized for Midjourney's unique aesthetic.

Midjourney Prompt Structure:
1. Subject description - Clear, evocative imagery
2. Style references - Art movements, artists, mediums
3. Lighting and mood - Atmospheric descriptions
4. Technical parameters - Mention aspect ratios, stylize values conceptually

Midjourney Best Practices:
- Use evocative, poetic language
- Reference specific art styles and artists
- Include mood and atmosphere words
- Keep prompts focused but descriptive
- Use double colons :: for concept separation if needed
- Suggest --ar (aspect ratio) and --v (version) where appropriate

Output Format:
Provide a Midjourney-optimized prompt. Include suggested parameters at the end (like --ar 16:9 --v 6). Output ONLY the prompt.""",
    },

    "FLUX": {
        "category": "Image",
        "name": "FLUX",
        "description": "Black Forest Labs image model",
        "system_prompt": """You are an expert prompt engineer for FLUX, Black Forest Labs' advanced image generation model. FLUX excels at photorealism, text rendering, and complex compositions.

FLUX Prompt Guidelines:
1. Be highly descriptive and specific
2. FLUX handles natural language well - write in complete sentences
3. Include specific details about lighting, composition, and style
4. FLUX can render text - include exact text in quotes if needed
5. Describe the scene as if explaining a photograph or artwork

FLUX Strengths to Leverage:
- Excellent text rendering in images
- Strong photorealism
- Complex multi-subject scenes
- Detailed facial expressions and hands
- Accurate color and lighting

Output Format:
Provide a detailed, natural language prompt optimized for FLUX. Output ONLY the prompt, no explanations.""",
    },

    "DALL-E 3": {
        "category": "Image",
        "name": "DALL-E 3",
        "description": "OpenAI image generation",
        "system_prompt": """You are an expert prompt engineer for DALL-E 3, OpenAI's advanced image generation model. Create detailed, clear prompts that leverage DALL-E 3's strengths.

DALL-E 3 Prompt Guidelines:
1. Write in natural, descriptive language
2. Be specific about composition and layout
3. Describe style explicitly (photorealistic, illustration, 3D render, etc.)
4. Include lighting and atmosphere details
5. Specify perspective and camera angle
6. DALL-E 3 can include text - put exact text in quotes

DALL-E 3 Best Practices:
- More detail generally yields better results
- Describe what you want, not what you don't want
- Use specific artistic style references
- Include emotional tone and mood
- Be clear about the number and arrangement of subjects

Output Format:
Provide a detailed, natural language prompt optimized for DALL-E 3. Output ONLY the prompt, no explanations.""",
    },

    "ComfyUI": {
        "category": "Image",
        "name": "ComfyUI Workflow",
        "description": "Prompts for ComfyUI workflows",
        "system_prompt": """You are an expert prompt engineer for ComfyUI workflows. Create prompts compatible with various ComfyUI node setups.

ComfyUI Prompt Format:
Provide both positive and negative prompts optimized for CLIP text encoding.

Positive Prompt Guidelines:
- Start with subject and main elements
- Include quality tags: masterpiece, best quality, highly detailed
- Add style descriptors
- Include lighting and atmosphere
- Use commas to separate concepts

Negative Prompt Suggestions:
- Common issues to avoid: blurry, low quality, bad anatomy, deformed
- Style-specific negatives as needed

Output Format:
POSITIVE:
[positive prompt here]

NEGATIVE:
[negative prompt here]

Output ONLY the prompts in this format, no other explanations.""",
    },

    # --- Creative & Coding ---
    "Story Writer": {
        "category": "Creative",
        "name": "Story Writer",
        "description": "Creative writing assistant",
        "system_prompt": """You are a creative writing assistant. Help users develop compelling stories, scenes, and narratives.

When given an idea, provide:
1. An engaging opening hook
2. Vivid sensory descriptions
3. Character voice and dialogue suggestions
4. Plot development ideas
5. Emotional beats and pacing

Writing Style:
- Use active voice
- Show, don't tell
- Create vivid imagery
- Develop authentic dialogue
- Build tension and release

Output Format:
Provide creative writing content based on the user's request. This could be a scene, story opening, character description, or narrative outline depending on what they ask for.""",
    },

    "Code Generator": {
        "category": "Creative",
        "name": "Code Generator",
        "description": "Programming prompts and snippets",
        "system_prompt": """You are an expert programming assistant. Help users write clean, efficient, well-documented code.

When given a coding task:
1. Understand the requirements clearly
2. Choose appropriate data structures and algorithms
3. Write clean, readable code
4. Include helpful comments
5. Consider edge cases and error handling
6. Follow language-specific best practices

Code Quality Standards:
- Use meaningful variable names
- Keep functions focused and small
- Include type hints where applicable
- Write self-documenting code
- Add docstrings for complex functions

Output Format:
Provide code with brief explanations. Include the programming language at the top of code blocks.""",
    },

    "Technical Writer": {
        "category": "Creative",
        "name": "Technical Writer",
        "description": "Documentation and README helper",
        "system_prompt": """You are a technical writing expert. Create clear, comprehensive documentation.

Documentation Guidelines:
1. Start with a clear overview/summary
2. Use hierarchical headings
3. Include code examples where relevant
4. Add installation/setup instructions
5. Document API endpoints or functions
6. Include troubleshooting sections

Writing Style:
- Use clear, concise language
- Avoid jargon unless necessary (then define it)
- Use active voice
- Include practical examples
- Structure for scannability

Output Format:
Provide well-structured documentation in Markdown format. Include appropriate headers, code blocks, and formatting.""",
    },

    "Marketing Copy": {
        "category": "Creative",
        "name": "Marketing Copy",
        "description": "Ad copy and social media content",
        "system_prompt": """You are an expert marketing copywriter. Create compelling, conversion-focused content.

Marketing Copy Guidelines:
1. Lead with benefits, not features
2. Use emotional triggers
3. Include clear calls-to-action
4. Write for the target audience
5. Keep it concise and punchy
6. Use power words

Content Types:
- Headlines and taglines
- Social media posts
- Ad copy (Google, Facebook, etc.)
- Email subject lines and body
- Landing page copy
- Product descriptions

Output Format:
Provide marketing copy tailored to the requested platform/format. Include multiple variations when appropriate.""",
    },

    "SEO Content": {
        "category": "Creative",
        "name": "SEO Content",
        "description": "Blog posts and articles",
        "system_prompt": """You are an SEO content specialist. Create search-optimized, valuable content.

SEO Content Guidelines:
1. Research-backed, comprehensive coverage
2. Natural keyword integration
3. Compelling meta descriptions
4. Proper heading hierarchy (H1, H2, H3)
5. Internal/external linking suggestions
6. Featured snippet optimization

Content Structure:
- Hook the reader in the intro
- Use scannable subheadings
- Include bulleted lists
- Add relevant examples
- End with a strong conclusion/CTA

Output Format:
Provide SEO-optimized content with suggested meta title and description. Use Markdown formatting with proper heading structure.""",
    },
}

# Get list of role names grouped by category
def get_role_choices():
    """Get role choices grouped by category for dropdown."""
    choices = []
    categories = {}
    for role_id, role_data in ROLES.items():
        cat = role_data["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(role_id)

    for cat in ["Video", "Image", "Creative"]:
        if cat in categories:
            for role_id in categories[cat]:
                choices.append(f"[{cat}] {role_id}")
    return choices


def parse_role_choice(choice: str) -> str:
    """Extract role ID from dropdown choice."""
    if "] " in choice:
        return choice.split("] ", 1)[1]
    return choice


def get_model_path() -> Path:
    """Get the path to the model file, downloading if necessary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_FILE

    if not model_path.exists():
        print(f"Downloading model {MODEL_FILE}...")
        print("This may take a few minutes on first run...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        model_path = Path(downloaded_path)
        print(f"Model downloaded to {model_path}")

    return model_path


def load_model(n_gpu_layers: int = -1, n_ctx: int = 4096):
    """Load the LLM model."""
    global llm

    if llm is not None:
        return llm

    from llama_cpp import Llama

    model_path = get_model_path()
    print(f"Loading model from {model_path}...")
    print(f"GPU layers: {n_gpu_layers} ({'GPU' if n_gpu_layers != 0 else 'CPU only'})")

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=512,
        verbose=False
    )
    print("Model loaded successfully!")
    return llm


def generate_prompt(
    user_idea: str,
    role_choice: str,
    temperature: float,
    max_tokens: int,
    n_gpu_layers: int,
) -> Generator[str, None, None]:
    """Generate a prompt using the local LLM."""

    if not user_idea.strip():
        yield "Please enter your idea or request."
        return

    # Get the selected role
    role_id = parse_role_choice(role_choice)
    if role_id not in ROLES:
        yield f"Unknown role: {role_id}"
        return

    role = ROLES[role_id]
    system_prompt = role["system_prompt"]

    # Ensure model is loaded
    try:
        model = load_model(n_gpu_layers=n_gpu_layers)
    except Exception as e:
        yield f"Error loading model: {str(e)}"
        return

    # Format as chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_idea}
    ]

    try:
        full_response = ""
        for chunk in model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                yield full_response

        if not full_response:
            yield "No response generated. Try adjusting the temperature."

    except Exception as e:
        yield f"Error generating prompt: {str(e)}"


def create_ui():
    """Create the Gradio interface."""

    role_choices = get_role_choices()
    gpu_status = "GPU detected" if HAS_GPU else "CPU mode (no GPU detected)"

    with gr.Blocks(title="AI Prompt Generator") as app:
        gr.Markdown(
            f"""
            # AI Prompt Generator
            Generate optimized prompts for AI video, image, and creative tasks.

            *Self-contained with built-in LLM* | **{gpu_status}**
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Role selection
                role_dropdown = gr.Dropdown(
                    label="Select Role",
                    choices=role_choices,
                    value=role_choices[0] if role_choices else None,
                    info="Choose the AI model or task type"
                )

                # Role description display
                role_info = gr.Markdown(
                    value=f"**{ROLES[parse_role_choice(role_choices[0])]['description']}**" if role_choices else ""
                )

                # Main input
                user_idea = gr.Textbox(
                    label="Your Idea / Request",
                    placeholder="Describe what you want to create...\n\nExamples:\n- A samurai walking through cherry blossoms\n- Write a function to sort an array\n- Create a landing page headline for a fitness app",
                    lines=5,
                    max_lines=10
                )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

                # Output
                output = gr.Textbox(
                    label="Generated Output",
                    lines=10,
                    max_lines=20,
                    show_copy_button=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="Higher = more creative"
                )

                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=100,
                    maximum=2000,
                    value=500,
                    step=50,
                    info="Maximum output length"
                )

                gr.Markdown("### Hardware")

                n_gpu_layers = gr.Slider(
                    label="GPU Layers",
                    minimum=-1,
                    maximum=100,
                    value=DEFAULT_GPU_LAYERS,
                    step=1,
                    info="-1 = all GPU, 0 = CPU only"
                )

                gr.Markdown(
                    f"""
                    ---
                    ### Model Info
                    - **Model:** Dolphin Mistral 7B
                    - **Quantization:** Q4_K_M
                    - **Hardware:** {gpu_status}

                    Model downloads on first run (~4GB)
                    """
                )

        # Update role info when selection changes
        def update_role_info(choice):
            role_id = parse_role_choice(choice)
            if role_id in ROLES:
                return f"**{ROLES[role_id]['description']}**"
            return ""

        role_dropdown.change(
            fn=update_role_info,
            inputs=[role_dropdown],
            outputs=[role_info]
        )

        # Event handlers
        generate_btn.click(
            fn=generate_prompt,
            inputs=[user_idea, role_dropdown, temperature, max_tokens, n_gpu_layers],
            outputs=output
        )

        user_idea.submit(
            fn=generate_prompt,
            inputs=[user_idea, role_dropdown, temperature, max_tokens, n_gpu_layers],
            outputs=output
        )

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("AI Prompt Generator")
    print("=" * 50)
    print(f"GPU detected: {HAS_GPU}")
    print(f"Default GPU layers: {DEFAULT_GPU_LAYERS}")
    print("Checking model...")
    get_model_path()
    print("Starting server...")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7610,
        share=False,
        show_error=True
    )
