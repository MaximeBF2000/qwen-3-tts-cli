"""Portable JSON profiles and locally copied audio; no pickle deserialization."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def profile_root():
    return Path(os.environ.get("QWEN3_TTS_HOME", Path.home() / ".local/share/qwen3-tts"))


def profile_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Profile names must use 1–64 letters, digits, underscores, or hyphens.")
    return profile_root() / "voices" / name


def load(name):
    path = profile_path(name)
    try:
        data = json.loads((path / "voice.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Voice profile {name!r} does not exist.") from None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"Unsupported voice profile: {name}")
    if not isinstance(data.get("audio"), str) or Path(data["audio"]).name != data["audio"]:
        raise ValueError(f"Invalid audio path in profile: {name}")
    data["ref_audio"] = str(path / data["audio"])
    return data


def save(name, audio, text, x_vector_only=False):
    destination = profile_path(name)
    if destination.exists():
        raise ValueError(f"Voice profile {name!r} already exists. Choose another name.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".voice-", dir=destination.parent))
    try:
        filename = "reference" + Path(audio).suffix.lower()
        shutil.copyfile(audio, temporary / filename)
        data = {"version": 1, "audio": filename, "ref_text": text, "x_vector_only": x_vector_only}
        (temporary / "voice.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def names():
    root = profile_root() / "voices"
    return sorted(
        p.parent.name for p in root.glob("*/voice.json") if not p.parent.name.startswith(".")
    )
