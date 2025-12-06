#!/usr/bin/env python3
"""
AI Prompt Generator for Wan2.x & Hunyuan Video

A self-contained Gradio web UI with selectable LLMs (based on GPU VRAM) for generating
optimized prompts for AI video generation (Wan2.1, Wan2.2, Hunyuan Video 1.5),
image generation (Stable Diffusion, FLUX, Midjourney, DALL-E 3), and creative tasks.

Features:
- Multiple LLM options from 0.5B to 14B parameters
- Auto-detection of GPU for optimal performance
- Specialized prompt engineering for each target AI model
- Streaming text generation
"""

import gradio as gr
import os
import subprocess
import threading
from pathlib import Path
from typing import Generator
from huggingface_hub import hf_hub_download

# Models directory
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

# =============================================================================
# MODEL CONFIGURATIONS BY VRAM
# =============================================================================

MODEL_CONFIGS = {
    "CPU Only (2-4GB RAM)": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "description": "Qwen2.5 0.5B Q8 - Lightweight, runs on CPU",
        "vram": "~1GB",
        "n_ctx": 4096,
    },
    "4GB VRAM (GTX 1650, RTX 3050)": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5 1.5B Q4_K_M - Good balance for low VRAM",
        "vram": "~2GB",
        "n_ctx": 4096,
    },
    "6GB VRAM (RTX 2060, RTX 3060)": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "description": "Qwen2.5 3B Q4_K_M - Great quality for 6GB cards",
        "vram": "~4GB",
        "n_ctx": 4096,
    },
    "8GB VRAM (RTX 3070, RTX 4060)": {
        "repo": "TheBloke/dolphin-2.6-mistral-7B-GGUF",
        "file": "dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "description": "Dolphin Mistral 7B Q4_K_M - Excellent prompt generation",
        "vram": "~6GB",
        "n_ctx": 4096,
    },
    "12GB VRAM (RTX 3060 12GB, RTX 4070)": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q6_K_L.gguf",
        "description": "Qwen2.5 7B Q6_K_L - High quality, better reasoning",
        "vram": "~10GB",
        "n_ctx": 8192,
    },
    "16GB+ VRAM (RTX 4080, RTX 4090)": {
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "description": "Qwen2.5 14B Q4_K_M - Premium quality prompts",
        "vram": "~12GB",
        "n_ctx": 8192,
    },
    "24GB+ VRAM (RTX 3090, RTX 4090)": {
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "file": "Qwen2.5-14B-Instruct-Q8_0.gguf",
        "description": "Qwen2.5 14B Q8 - Maximum quality",
        "vram": "~18GB",
        "n_ctx": 8192,
    },
}

# Global model instance and current model tracking
llm = None
current_model_key = None
unload_timer = None
UNLOAD_DELAY_SECONDS = 10


def detect_gpu() -> tuple[bool, int, str]:
    """Detect if CUDA GPU is available and return VRAM in MB.

    Returns:
        tuple: (has_gpu, vram_mb, gpu_name)
    """
    try:
        # Query GPU memory and name
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,name', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]  # First GPU
            parts = line.split(', ')
            vram_mb = int(parts[0].strip())
            gpu_name = parts[1].strip() if len(parts) > 1 else "Unknown GPU"
            return True, vram_mb, gpu_name
        return False, 0, ""
    except Exception:
        return False, 0, ""


def select_model_by_vram(vram_mb: int) -> str:
    """Select the best model based on available VRAM."""
    vram_gb = vram_mb / 1024

    if vram_gb >= 20:
        return "24GB+ VRAM (RTX 3090, RTX 4090)"
    elif vram_gb >= 14:
        return "16GB+ VRAM (RTX 4080, RTX 4090)"
    elif vram_gb >= 10:
        return "12GB VRAM (RTX 3060 12GB, RTX 4070)"
    elif vram_gb >= 7:
        return "8GB VRAM (RTX 3070, RTX 4060)"
    elif vram_gb >= 5:
        return "6GB VRAM (RTX 2060, RTX 3060)"
    elif vram_gb >= 3:
        return "4GB VRAM (GTX 1650, RTX 3050)"
    else:
        return "CPU Only (2-4GB RAM)"


# Auto-detect GPU on startup
HAS_GPU, GPU_VRAM_MB, GPU_NAME = detect_gpu()
DEFAULT_GPU_LAYERS = -1 if HAS_GPU else 0

# Auto-select model based on VRAM
if HAS_GPU and GPU_VRAM_MB > 0:
    DEFAULT_MODEL = select_model_by_vram(GPU_VRAM_MB)
else:
    DEFAULT_MODEL = "CPU Only (2-4GB RAM)"


# =============================================================================
# ROLE DEFINITIONS
# =============================================================================

ROLES = {
    # --- Video Generation ---
    "Wan2.1": {
        "category": "Video",
        "name": "Wan2.1 Video",
        "description": "Open-source text-to-video model with cinematic quality",
        "system_prompt": """You are an expert prompt engineer specializing in Wan2.1, a powerful open-source text-to-video AI model. Your task is to transform user ideas into highly effective Wan2.1 prompts.

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

    "Wan2.2": {
        "category": "Video",
        "name": "Wan2.2 Video",
        "description": "Latest Wan model with improved motion, physics and quality",
        "system_prompt": """You are an expert prompt engineer specializing in Wan2.2, the latest open-source text-to-video AI model. Wan2.2 features significant improvements over Wan2.1 including better motion coherence, enhanced visual quality, improved temporal consistency, and stronger prompt adherence.

Wan2.2 Prompt Structure:
1. Shot type - Camera perspective (extreme close-up, close-up, medium shot, wide shot, establishing shot, aerial view, POV, over-the-shoulder, tracking shot, dutch angle)
2. Subject description - Highly detailed description of the main subject (physical features, clothing details, textures, colors, expressions, body language)
3. Action/Motion - Precise motion description with temporal flow (use present continuous tense, describe motion phases)
4. Environment/Setting - Rich environmental details (location type, architectural elements, natural features, time of day, season, weather conditions)
5. Lighting - Specific lighting setup (golden hour, blue hour, harsh noon sun, overcast diffused, neon glow, rim lighting, chiaroscuro, volumetric rays)
6. Style/Aesthetic - Visual style with references (cinematic 35mm film, anamorphic lens, IMAX quality, documentary realism, anime/animation style, surrealist, noir)
7. Camera movement - Dynamic camera description (slow dolly in, sweeping crane shot, steadicam follow, whip pan, push in, pull out, orbital movement)
8. Atmosphere/Mood - Emotional tone and ambiance (tension, serenity, mystery, joy, melancholy)

Wan2.2 Enhancements to Leverage:
- Superior motion physics and natural movement flow
- Better handling of complex multi-subject interactions
- Improved face consistency and expressions throughout the clip
- Enhanced text rendering capabilities
- Better understanding of spatial relationships
- More accurate physics simulation (cloth, hair, water, particles)

Best Practices:
- Use present continuous tense for all actions ("A dancer is gracefully spinning" not "A dancer spins")
- Layer motion descriptions temporally (what happens first, then, finally)
- Be extremely specific about textures, materials, and surface qualities
- Include micro-movements and secondary motion (hair flowing, clothes rustling, leaves drifting)
- Specify the emotional arc or mood progression if applicable
- Keep prompts between 150-250 words for optimal detail without confusion
- Describe lighting direction and quality, not just type

Output Format:
Provide ONE cohesive, detailed prompt as a flowing paragraph. Front-load the most critical visual elements. Output ONLY the prompt, no explanations or preamble.""",
    },

    "Hunyuan Video": {
        "category": "Video",
        "name": "Hunyuan Video",
        "description": "Open-source text-to-video with natural motion",
        "system_prompt": """You are an expert prompt engineer for Hunyuan Video, an advanced open-source text-to-video AI model. Transform user ideas into optimized Hunyuan prompts.

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

    "Hunyuan Video 1.5": {
        "category": "Video",
        "name": "Hunyuan Video 1.5",
        "description": "Advanced video model with image-to-video and extended duration",
        "system_prompt": """You are an expert prompt engineer for Hunyuan Video 1.5, a state-of-the-art open-source text-to-video and image-to-video AI model. This version features major improvements including longer video generation, image-to-video capabilities, better motion control, and enhanced visual fidelity.

Hunyuan Video 1.5 Capabilities:
- Text-to-video generation with superior temporal consistency
- Image-to-video animation (bring still images to life)
- Extended video duration support
- Improved motion dynamics and physics
- Better facial expression and body movement handling
- Enhanced lighting and atmosphere rendering

Prompt Structure for Text-to-Video:
1. Subject - Detailed description of main subject(s) with specific visual attributes (age, appearance, clothing style, colors, textures)
2. Action/Movement - Precise motion description using present continuous tense, include subtle movements
3. Environment - Rich setting details (interior/exterior, architecture, natural elements, props)
4. Time & Lighting - Specific time of day and lighting conditions (dawn light streaming through windows, neon-lit rainy street, soft studio lighting)
5. Camera - Shot type and movement (tracking shot following subject, slow push in, static wide shot, handheld documentary style)
6. Style - Visual aesthetic (photorealistic, cinematic film look, stylized, specific film/director references)
7. Atmosphere - Mood and emotional tone (tense, dreamlike, energetic, contemplative)

Image-to-Video Prompt Tips:
When animating a reference image, focus on:
- What specific movements should occur
- Direction and speed of motion
- What elements should remain static vs. animate
- Camera movement relative to the scene
- Environmental effects to add (wind, particles, lighting changes)

Best Practices:
- Use natural, flowing language - Hunyuan 1.5 understands context well
- Be specific about motion speed and intensity (slowly, rapidly, gently, dramatically)
- Describe cause-and-effect motion (wind causing hair to flow, footsteps creating ripples)
- Include ambient motion (background elements moving naturally)
- Specify emotional expressions and their changes throughout the scene
- Layer details from most important to supporting elements
- Aim for 100-200 words for optimal prompt interpretation
- For complex scenes, break down the temporal sequence of events

Motion Quality Keywords:
- Smooth, fluid, graceful (for elegant movement)
- Dynamic, energetic, rapid (for action scenes)
- Subtle, gentle, delicate (for nuanced motion)
- Cinematic, dramatic, sweeping (for epic shots)

Output Format:
Provide ONE cohesive prompt as a detailed paragraph. For image-to-video requests, assume the user will provide the reference image separately and focus your prompt on describing the desired animation. Output ONLY the prompt, no explanations.""",
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


def check_model_exists(model_key: str) -> bool:
    """Check if model file exists locally."""
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])
    model_path = MODELS_DIR / config["file"]
    return model_path.exists()


def get_model_path(model_key: str, progress_callback=None) -> Path:
    """Get the path to the model file, downloading if necessary."""
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])
    model_file = config["file"]
    model_repo = config["repo"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        print(f"Downloading model {model_file} from {model_repo}...")
        print("This may take a few minutes on first run...")
        if progress_callback:
            progress_callback(f"⬇️ Downloading {config['description']}...\n\nThis may take a few minutes on first run.")
        downloaded_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        model_path = Path(downloaded_path)
        print(f"Model downloaded to {model_path}")
        if progress_callback:
            progress_callback(f"✅ Download complete! Loading model...")

    return model_path


def unload_model():
    """Unload the current model to free memory."""
    global llm, current_model_key
    if llm is not None:
        model_name = current_model_key
        del llm
        llm = None
        current_model_key = None
        # Try to free GPU memory
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        print(f"Model unloaded: {model_name}")


def cancel_unload_timer():
    """Cancel any pending auto-unload timer."""
    global unload_timer
    if unload_timer is not None:
        unload_timer.cancel()
        unload_timer = None


def schedule_unload():
    """Schedule the model to be unloaded after UNLOAD_DELAY_SECONDS."""
    global unload_timer
    cancel_unload_timer()
    unload_timer = threading.Timer(UNLOAD_DELAY_SECONDS, auto_unload_model)
    unload_timer.daemon = True
    unload_timer.start()
    print(f"Model will auto-unload in {UNLOAD_DELAY_SECONDS} seconds...")


def auto_unload_model():
    """Auto-unload callback triggered by timer."""
    global unload_timer
    unload_timer = None
    if llm is not None:
        print(f"Auto-unloading model after {UNLOAD_DELAY_SECONDS}s of inactivity...")
        unload_model()


def load_model(model_key: str, n_gpu_layers: int = -1, progress_callback=None):
    """Load the LLM model."""
    global llm, current_model_key

    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])

    # Check if we need to switch models
    if llm is not None and current_model_key == model_key:
        return llm

    # Unload existing model if switching
    if llm is not None and current_model_key != model_key:
        print(f"Switching model from {current_model_key} to {model_key}...")
        if progress_callback:
            progress_callback(f"🔄 Switching to {config['description']}...")
        unload_model()

    from llama_cpp import Llama

    model_path = get_model_path(model_key, progress_callback)
    n_ctx = config.get("n_ctx", 4096)

    print(f"Loading model: {config['description']}")
    print(f"Model path: {model_path}")
    print(f"GPU layers: {n_gpu_layers} ({'GPU' if n_gpu_layers != 0 else 'CPU only'})")
    print(f"Context size: {n_ctx}")

    if progress_callback:
        progress_callback(f"🔄 Loading {config['description']}...")

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=512,
        verbose=False
    )
    current_model_key = model_key
    print("Model loaded successfully!")
    return llm


def generate_prompt(
    user_idea: str,
    role_choice: str,
    model_choice: str,
    temperature: float,
    max_tokens: int,
    n_gpu_layers: int,
) -> Generator[str, None, None]:
    """Generate a prompt using the local LLM."""

    if not user_idea.strip():
        yield "Please enter your idea or request."
        return

    # Cancel any pending unload since we're about to use the model
    cancel_unload_timer()

    # Get the selected role
    role_id = parse_role_choice(role_choice)
    if role_id not in ROLES:
        yield f"Unknown role: {role_id}"
        return

    role = ROLES[role_id]
    system_prompt = role["system_prompt"]

    # Status tracking for UI feedback
    status_message = [None]  # Use list to allow modification in nested function

    def update_status(msg):
        status_message[0] = msg

    # Check if model needs downloading
    needs_download = not check_model_exists(model_choice)
    if needs_download:
        config = MODEL_CONFIGS.get(model_choice, MODEL_CONFIGS[DEFAULT_MODEL])
        yield f"⬇️ Downloading {config['description']}...\n\nThis is a one-time download. Please wait..."

    # Ensure model is loaded
    try:
        model = load_model(
            model_key=model_choice,
            n_gpu_layers=n_gpu_layers,
            progress_callback=update_status
        )
        if needs_download:
            yield "✅ Model ready! Generating prompt..."
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
        chunk_count = 0
        for chunk in model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            chunk_count += 1
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                yield full_response

        if not full_response:
            hints = []
            if n_gpu_layers != 0:
                hints.append("Set GPU Layers to 0 (CPU mode)")
            hints.append("Try increasing temperature")
            hints.append("Try a shorter input")
            yield f"No response generated. Try:\n- " + "\n- ".join(hints)

        # Schedule auto-unload after generation completes
        schedule_unload()

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "memory" in error_msg or "cuda" in error_msg or "gpu" in error_msg:
            yield f"GPU/Memory error: {str(e)}\n\nTry setting GPU Layers to 0 for CPU-only mode."
        else:
            yield f"Runtime error: {str(e)}"
    except Exception as e:
        yield f"Error generating prompt: {str(e)}\n\nIf this persists, try restarting the application."


def create_ui():
    """Create the Gradio interface."""

    role_choices = get_role_choices()

    # Build GPU status string
    if HAS_GPU and GPU_VRAM_MB > 0:
        vram_gb = GPU_VRAM_MB / 1024
        gpu_status = f"{GPU_NAME} ({vram_gb:.0f}GB VRAM)"
    elif HAS_GPU:
        gpu_status = "GPU detected"
    else:
        gpu_status = "CPU mode (no GPU detected)"

    with gr.Blocks(title="AI Prompt Generator - Wan2.x & Hunyuan") as app:
        gr.Markdown(
            f"""
            # AI Prompt Generator
            Generate optimized prompts for **Wan2.1/2.2**, **Hunyuan Video 1.5**, Stable Diffusion, FLUX, and more.

            *Select your GPU tier for optimal performance* | **{gpu_status}**
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Role selection
                role_dropdown = gr.Dropdown(
                    label="Target AI Model",
                    choices=role_choices,
                    value=role_choices[0] if role_choices else None,
                    info="Select the AI model you're generating prompts for"
                )

                # Role description display
                role_info = gr.Markdown(
                    value=f"**{ROLES[parse_role_choice(role_choices[0])]['description']}**" if role_choices else ""
                )

                # Main input
                user_idea = gr.Textbox(
                    label="Your Idea / Request",
                    placeholder="Describe what you want to create, or click an example below...",
                    lines=5,
                    max_lines=10
                )

                # Example buttons
                gr.Markdown("**Quick Examples:**")
                with gr.Row():
                    ex_btn1 = gr.Button("Samurai in Cherry Blossoms", size="sm")
                    ex_btn2 = gr.Button("Timelapse Flower Bloom", size="sm")
                    ex_btn3 = gr.Button("Ocean Waves Aerial", size="sm")
                with gr.Row():
                    ex_btn4 = gr.Button("Cyberpunk Portrait", size="sm")
                    ex_btn5 = gr.Button("Cozy Cabin Snow", size="sm")
                    ex_btn6 = gr.Button("Astronaut on Mars", size="sm")

                generate_btn = gr.Button("Generate Prompt", variant="primary", size="lg")

                # Output
                output = gr.Textbox(
                    label="Generated Prompt",
                    lines=10,
                    max_lines=20,
                    show_copy_button=True,
                    info="Copy this prompt to use with your AI model"
                )

            with gr.Column(scale=1):
                # Show auto-detection status
                if HAS_GPU and GPU_VRAM_MB > 0:
                    vram_gb = GPU_VRAM_MB / 1024
                    gr.Markdown(f"### LLM for Prompt Generation\n*Auto-detected: {vram_gb:.0f}GB VRAM*")
                else:
                    gr.Markdown("### LLM for Prompt Generation\n*No GPU detected - using CPU*")

                model_choices = list(MODEL_CONFIGS.keys())
                model_dropdown = gr.Dropdown(
                    label="Select by Your GPU VRAM",
                    choices=model_choices,
                    value=DEFAULT_MODEL,
                    info="Auto-selected based on detected VRAM" if HAS_GPU else "Select manually or use CPU model"
                )

                # Model description display
                model_info = gr.Markdown(
                    value=f"**{MODEL_CONFIGS[DEFAULT_MODEL]['description']}**\n\nVRAM usage: {MODEL_CONFIGS[DEFAULT_MODEL]['vram']}"
                )

                gr.Markdown("### Output Settings")

                temperature = gr.Slider(
                    label="Creativity (Temperature)",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="0.3-0.5 precise, 0.7-1.0 creative, 1.0+ experimental"
                )

                max_tokens = gr.Slider(
                    label="Max Length (Tokens)",
                    minimum=100,
                    maximum=2000,
                    value=256,
                    step=50,
                    info="Video prompts: 150-300, Image prompts: 75-150"
                )

                gr.Markdown("### Advanced")

                n_gpu_layers = gr.Slider(
                    label="GPU Layers",
                    minimum=-1,
                    maximum=100,
                    value=DEFAULT_GPU_LAYERS,
                    step=1,
                    info="-1 = all layers on GPU, 0 = CPU only"
                )

                gr.Markdown(
                    f"""
                    ---
                    **Status:** {gpu_status}

                    *Models auto-download on first use.*
                    *Changing model will free current memory.*
                    """
                )

        # Update role info when selection changes
        def update_role_info(choice):
            role_id = parse_role_choice(choice)
            if role_id in ROLES:
                return f"**{ROLES[role_id]['description']}**"
            return ""

        # Update model info when selection changes
        def update_model_info(choice):
            if choice in MODEL_CONFIGS:
                config = MODEL_CONFIGS[choice]
                return f"**{config['description']}**\n\nVRAM usage: {config['vram']}"
            return ""

        role_dropdown.change(
            fn=update_role_info,
            inputs=[role_dropdown],
            outputs=[role_info]
        )

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )

        # Example button handlers
        example_prompts = {
            "ex1": "A lone samurai walking slowly through a path of falling cherry blossoms at golden hour sunset, katana at his side, petals swirling in the gentle breeze",
            "ex2": "Macro timelapse of a delicate flower bud slowly opening and blooming in a sunlit garden, dewdrops glistening on petals, soft bokeh background",
            "ex3": "Cinematic aerial drone shot of powerful turquoise ocean waves crashing against dramatic rocky cliffs, white foam spray, golden hour lighting",
            "ex4": "Close-up portrait of a cyberpunk warrior with glowing neon tattoos, rain-soaked face, reflections of holographic billboards, moody night scene",
            "ex5": "Cozy wooden cabin nestled in snowy mountains at twilight, warm light glowing from windows, smoke rising from chimney, fresh snowfall",
            "ex6": "An astronaut in a detailed spacesuit walking across the rusty red Martian surface, Earth visible in the distant sky, dramatic shadows",
        }

        ex_btn1.click(fn=lambda: example_prompts["ex1"], outputs=user_idea)
        ex_btn2.click(fn=lambda: example_prompts["ex2"], outputs=user_idea)
        ex_btn3.click(fn=lambda: example_prompts["ex3"], outputs=user_idea)
        ex_btn4.click(fn=lambda: example_prompts["ex4"], outputs=user_idea)
        ex_btn5.click(fn=lambda: example_prompts["ex5"], outputs=user_idea)
        ex_btn6.click(fn=lambda: example_prompts["ex6"], outputs=user_idea)

        # Event handlers
        generate_btn.click(
            fn=generate_prompt,
            inputs=[user_idea, role_dropdown, model_dropdown, temperature, max_tokens, n_gpu_layers],
            outputs=output
        )

        user_idea.submit(
            fn=generate_prompt,
            inputs=[user_idea, role_dropdown, model_dropdown, temperature, max_tokens, n_gpu_layers],
            outputs=output
        )

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("AI Prompt Generator")
    print("=" * 50)
    if HAS_GPU and GPU_VRAM_MB > 0:
        vram_gb = GPU_VRAM_MB / 1024
        print(f"GPU: {GPU_NAME}")
        print(f"VRAM: {vram_gb:.1f} GB ({GPU_VRAM_MB} MB)")
    else:
        print("GPU: Not detected (CPU mode)")
    print(f"Auto-selected model: {DEFAULT_MODEL}")
    print(f"Available models: {len(MODEL_CONFIGS)}")
    print("Starting server...")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7610,
        share=False,
        show_error=True
    )
