"""Starter examples, one set per role category.

The example buttons used to show video ideas regardless of the selected target,
which made them useless for audio, 3D and creative roles. Each category now
carries its own set.
"""

from promptmill.domain.entities.role import RoleCategory

# Six (label, prompt) pairs per category; the UI renders them as two rows.
type ExamplePair = tuple[str, str]

EXAMPLES_BY_CATEGORY: dict[RoleCategory, tuple[ExamplePair, ...]] = {
    RoleCategory.VIDEO: (
        (
            "Samurai in Blossoms",
            "A lone samurai walking slowly through a path of falling cherry blossoms at golden hour sunset, katana at his side, petals swirling in the gentle breeze",
        ),
        (
            "Timelapse Bloom",
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
    ),
    RoleCategory.IMAGE: (
        (
            "Studio Portrait",
            "Studio portrait of an elderly ceramicist, clay dust on her hands and apron, soft window light from the left, shallow depth of field",
        ),
        (
            "Product Shot",
            "Product shot of a matte black mechanical watch on wet slate, single hard light source, water beads catching the highlight, deep shadows",
        ),
        (
            "Isometric Room",
            "Isometric cutaway of a tiny book repair workshop, warm lamps, stacked paper, tools on a pegboard, muted retro palette",
        ),
        (
            "Fantasy Landscape",
            "A vast floating archipelago above a sea of clouds, waterfalls pouring off the island edges, distant airships, late afternoon light",
        ),
        (
            "Poster Typography",
            "Minimal concert poster with the text 'NIGHT SHIFT' in bold condensed type, risograph texture, two-color print, heavy grain",
        ),
        (
            "Food Close-up",
            "Overhead close-up of a cast iron pan with caramelized onions, steam rising, rough linen underneath, moody side lighting",
        ),
    ),
    RoleCategory.AUDIO: (
        (
            "Lo-fi Study Beat",
            "A relaxed lo-fi hip hop track for studying, dusty Rhodes chords, brushed drums, vinyl crackle, 78 BPM, warm and unhurried",
        ),
        (
            "Cinematic Trailer",
            "An epic orchestral trailer cue building from a lone piano motif to full brass and taiko drums, 140 BPM, D minor",
        ),
        (
            "Indie Folk Song",
            "An indie folk song about leaving a coastal town, fingerpicked acoustic guitar, close-mic'd female vocal, brushed snare, gentle swell",
        ),
        (
            "Synthwave Drive",
            "Instrumental synthwave for a night drive, analog arpeggios, gated reverb snare, fretless bass, 110 BPM, A minor",
        ),
        (
            "Podcast Intro",
            "A 15 second podcast intro bed, curious and modern, plucked synth, light percussion, resolves cleanly for a voice-over",
        ),
        (
            "Rain Ambience",
            "Heavy rain on a metal roof with distant thunder, recorded from inside a small wooden shed, no music",
        ),
    ),
    RoleCategory.THREE_D: (
        (
            "Sci-fi Helmet",
            "A hard-surface sci-fi pilot helmet, brushed titanium shell, amber tinted visor, worn rubber seals, game-ready, neutral upright pose",
        ),
        (
            "Stylized Tree",
            "A stylized low-poly oak tree with chunky bark and flat-shaded foliage clusters, hand-painted look, upright",
        ),
        (
            "Antique Chair",
            "An antique wooden armchair with carved legs, faded green velvet upholstery, scuffed varnish, realistic PBR, upright",
        ),
        (
            "Robot Character",
            "A small bipedal service robot, rounded white plastic panels, exposed cable joints, single blue lens eye, A-pose for rigging",
        ),
        (
            "Fantasy Weapon",
            "A curved fantasy dagger, damascus steel blade, wrapped leather grip, tarnished brass pommel, game-ready, blade pointing up",
        ),
        (
            "Coffee Machine",
            "A retro espresso machine, polished chrome body, bakelite handles, brass portafilter, visible rivets, product-render quality",
        ),
    ),
    RoleCategory.CREATIVE: (
        (
            "Product Launch",
            "Announce a self-hosted analytics tool for small teams that stores no personal data, aimed at developers tired of cookie banners",
        ),
        (
            "Short Story",
            "A lighthouse keeper starts receiving letters addressed to someone who died forty years ago, and the handwriting is their own",
        ),
        (
            "Technical Doc",
            "Document a REST endpoint that accepts a batch of image URLs and returns per-image moderation scores, including error cases",
        ),
        (
            "Landing Page",
            "Landing page copy for a bike repair subscription in Berlin, two visits a year, pickup from your door, 12 euro a month",
        ),
        (
            "Conference Talk",
            "A 25 minute talk pitch about why our team deleted 40 percent of its microservices and what it cost us to find that out",
        ),
        (
            "Release Notes",
            "Release notes for version 2.4: faster cold starts, a breaking change to the config format, and three bug fixes",
        ),
    ),
}


def examples_for(category: RoleCategory) -> tuple[ExamplePair, ...]:
    """Get the example set for a category.

    Args:
        category: The role category to look up.

    Returns:
        Six (label, prompt) pairs; falls back to the creative set.
    """
    return EXAMPLES_BY_CATEGORY.get(category, EXAMPLES_BY_CATEGORY[RoleCategory.CREATIVE])
