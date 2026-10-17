"""Lazy inference imports keep help, validation, and profile commands lightweight."""

import gc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def configure_environment(offline=False, cache_dir=None):
    # Must happen before importing HF / Transformers: they cache these settings at import time.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if cache_dir:
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir).expanduser().resolve())


def resolve_model(model, offline=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if str(model).startswith(("/", ".", "~")):
        raise ValueError(f"Local model directory does not exist: {model}")
    from huggingface_hub import snapshot_download

    # Resolve everything once, including speech_tokenizer/, then give upstream a local path.
    # Its processor loader otherwise ignores cache_dir / local_files_only kwargs.
    return snapshot_download(repo_id=model, local_files_only=offline)


def download(model, offline=False):
    return resolve_model(model, offline)


class Runtime:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.key = None
        self.cpu_fallback = False

    def device_settings(self, torch):
        device = self.args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.cpu_fallback:
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
        dtype = self.args.dtype
        if dtype == "auto":
            dtype = "float32" if device in ("cpu", "mps") else "float16"
        if self.args.attention == "flash_attention_2" and (
            not device.startswith("cuda") or dtype == "float32"
        ):
            raise ValueError("FlashAttention 2 requires CUDA and float16/bfloat16.")
        return device, getattr(torch, dtype)

    def load(self, model_id):
        import torch
        from qwen_tts import Qwen3TTSModel

        device, dtype = self.device_settings(torch)
        key = (model_id, device, dtype)
        if key != self.key:
            self.model = None
            self.key = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            location = resolve_model(model_id, self.args.offline)
            print(f"Loading {model_id} on {device} ({dtype})…", file=sys.stderr)
            self.model = Qwen3TTSModel.from_pretrained(
                location,
                device_map=device,
                dtype=dtype,
                attn_implementation=self.args.attention,
            )
            self.key = key
        return self.model

    def generate(self, request):
        import torch

        def run():
            model = self.load(request["checkpoint"])
            if request.get("seed") is not None:
                torch.manual_seed(request["seed"])
            kwargs = {"text": request["text"], "language": request["language"]}
            for name in ("temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"):
                if request.get(name) is not None:
                    kwargs[name] = request[name]
            mode = request["mode"]
            if mode in ("custom", "design"):
                kwargs["instruct"] = request.get("instruct") or ""
            if mode == "custom":
                kwargs["speaker"] = request["speaker"]
            elif mode == "clone":
                kwargs.update(
                    ref_audio=request["ref_audio"],
                    ref_text=request.get("ref_text"),
                    x_vector_only_mode=request.get("x_vector_only", False),
                )
            method = {
                "custom": "generate_custom_voice",
                "design": "generate_voice_design",
                "clone": "generate_voice_clone",
            }[mode]
            with torch.inference_mode():
                waves, rate = getattr(model, method)(**kwargs)
            if len(waves) != 1 or len(waves[0]) == 0:
                raise ValueError("The model returned no audio.")
            return waves[0], rate

        try:
            return run()
        except (RuntimeError, NotImplementedError) as error:
            if (
                self.args.device != "auto"
                or self.cpu_fallback
                or not torch.backends.mps.is_available()
            ):
                raise
            print(f"MPS failed ({error}); retrying on CPU…", file=sys.stderr)
            self.cpu_fallback = True
        # Leave the exception handler first so its traceback releases MPS tensors.
        return run()


def write_audio(path, audio, rate, force=False):
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination, then publish a complete WAV atomically.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
            sf.write(stream, audio, rate, format="WAV", subtype="PCM_16")
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic exclusive creation; never replace an existing file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def play(path):
    if sys.platform == "darwin" and shutil.which("afplay"):
        subprocess.run(["afplay", str(path)], check=True)
    else:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ValueError("Playback requires: pip install 'qwen3-tts-cli[playback]'") from None
        audio, rate = sf.read(path, dtype="float32")
        sd.play(audio, rate)
        sd.wait()
