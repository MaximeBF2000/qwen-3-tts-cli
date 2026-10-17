"""Published upstream checkpoint and language catalog."""

MODELS = {
    "design": {"1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"},
    "clone": {size: f"Qwen/Qwen3-TTS-12Hz-{size}-Base" for size in ("0.6B", "1.7B")},
    "custom": {size: f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice" for size in ("0.6B", "1.7B")},
}
LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)
SPEAKERS = {
    "Vivian": "Bright young female voice; Chinese",
    "Serena": "Warm young female voice; Chinese",
    "Uncle_Fu": "Low, mellow male voice; Chinese",
    "Dylan": "Youthful male voice; Beijing Chinese",
    "Eric": "Lively male voice; Sichuan Chinese",
    "Ryan": "Rhythmic male voice; English",
    "Aiden": "Clear American male voice; English",
    "Ono_Anna": "Playful female voice; Japanese",
    "Sohee": "Warm female voice; Korean",
}


def canonical(value, choices, label):
    for choice in choices:
        if value.casefold() == choice.casefold():
            return choice
    raise ValueError(f"Unknown {label} {value!r}. Choose from: {', '.join(choices)}")


def checkpoint(mode, size, model=None):
    if model:
        return model
    if size not in MODELS[mode]:
        raise ValueError("VoiceDesign only has a 1.7B checkpoint; use --size 1.7B.")
    return MODELS[mode][size]
