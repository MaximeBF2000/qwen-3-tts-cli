"""Command parsing, input validation, and local workflows."""

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
from pathlib import Path

from . import __version__, profiles, runtime
from .catalog import LANGUAGES, MODELS, SPEAKERS, canonical, checkpoint


def model_options(parser):
    parser.add_argument(
        "--size",
        choices=("0.6B", "1.7B"),
        default="1.7B",
        help="Checkpoint size (default: 1.7B; design requires 1.7B).",
    )
    parser.add_argument("--model", help="Override with a local model directory or Hugging Face ID.")


def environment_options(parser):
    parser.add_argument(
        "--offline", action="store_true", help="Use local files/cache; disable Hub access."
    )
    parser.add_argument("--cache-dir", help="Override the Hugging Face model cache directory.")


def runtime_options(parser):
    environment_options(parser)
    parser.add_argument("--device", default="auto", help="auto (MPS → CPU), cpu, mps, or cuda:0.")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing output WAV files.")
    parser.add_argument("--play", action="store_true", help="Play audio after writing it.")


def generation_options(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to speak. Otherwise use --text-file or piped stdin.")
    source.add_argument("--text-file", help="UTF-8 text file, or - for stdin.")
    parser.add_argument(
        "-o", "--output", default="output.wav", help="Output WAV (default: output.wav)."
    )
    parser.add_argument("--language", default="Auto", help="Language name (default: Auto).")
    parser.add_argument("--seed", type=int, help="Random seed; results can vary by device.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    model_options(parser)
    runtime_options(parser)


def reference_options(parser):
    parser.add_argument("--ref-audio", help="Local reference audio file.")
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument("--ref-text", help="Exact transcript of the reference audio.")
    transcript.add_argument("--ref-text-file", help="UTF-8 reference transcript file.")
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Clone speaker embedding without a transcript (lower fidelity).",
    )


def parser():
    root = argparse.ArgumentParser(
        prog="qwen3-tts",
        description="Local Qwen3-TTS: design voices, clone audio, or use preset speakers.",
        epilog="Start with: qwen3-tts custom --text 'Hello!' --speaker Ryan -o hello.wav",
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for mode, help_text in (
        ("design", "Describe a voice in natural language."),
        ("clone", "Clone a local reference clip or saved voice."),
        ("custom", "Speak with a built-in speaker."),
    ):
        command = commands.add_parser(mode, help=help_text, description=help_text)
        generation_options(command)
        if mode in ("design", "custom"):
            command.add_argument(
                "--instruct",
                required=mode == "design",
                help="Voice description (design) or delivery/style instruction (custom).",
            )
        if mode == "custom":
            command.add_argument(
                "--speaker", default="Ryan", help="Preset speaker (default: Ryan)."
            )
        if mode == "clone":
            reference_options(command)
            command.add_argument("--voice", help="Saved voice profile name.")
        if mode in ("design", "clone"):
            command.add_argument(
                "--save-voice", help="Save a reusable local reference voice profile."
            )
    batch = commands.add_parser("batch", help="Generate jobs from a JSONL manifest.")
    batch.add_argument(
        "manifest", type=Path, help="One JSON object per line; paths relative to manifest."
    )
    runtime_options(batch)
    download = commands.add_parser(
        "download", help="Download a complete checkpoint for offline use."
    )
    download.add_argument("mode", choices=MODELS)
    model_options(download)
    environment_options(download)
    commands.add_parser("models", help="List published checkpoints without loading them.")
    commands.add_parser("speakers", help="List built-in CustomVoice speakers.")
    commands.add_parser("languages", help="List supported languages.")
    commands.add_parser("doctor", help="Report installed runtime and available devices.")
    voices = commands.add_parser("voices", help="Manage local reusable voice profiles.")
    sub = voices.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List saved profiles.")
    add = sub.add_parser("add", help="Copy a reference clip into a named profile.")
    add.add_argument("name")
    reference_options(add)
    for action in ("show", "delete"):
        item = sub.add_parser(action)
        item.add_argument("name")
    return root


def read_text(data, allow_stdin=True):
    if data.get("text") is not None and data.get("text_file") is not None:
        raise ValueError("Use text or text_file, not both.")
    value = data.get("text")
    if value is None:
        filename = data.get("text_file")
        if filename == "-" or (filename is None and allow_stdin and not sys.stdin.isatty()):
            if not allow_stdin:
                raise ValueError("Batch jobs cannot read stdin.")
            value = sys.stdin.read()
        elif filename:
            value = Path(filename).read_text(encoding="utf-8")
        else:
            raise ValueError("Provide --text, --text-file, or piped text on stdin.")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Text must be a non-empty string.")
    return value.strip()


def reference(data):
    if "x_vector_only" in data and type(data["x_vector_only"]) is not bool:
        raise ValueError("x_vector_only must be a JSON boolean.")
    if data.get("voice"):
        if any(data.get(k) for k in ("ref_audio", "ref_text", "ref_text_file", "x_vector_only")):
            raise ValueError("Use --voice by itself, without reference options.")
        data.update(profiles.load(data["voice"]))
    if data.get("ref_text") is not None and data.get("ref_text_file") is not None:
        raise ValueError("Use ref_text or ref_text_file, not both.")
    if data.get("ref_text_file"):
        data["ref_text"] = Path(data["ref_text_file"]).read_text(encoding="utf-8").strip()
    audio = data.get("ref_audio")
    if not isinstance(audio, str) or not Path(audio).is_file():
        raise ValueError("Provide an existing local --ref-audio file or --voice profile.")
    if not data.get("x_vector_only") and not (
        isinstance(data.get("ref_text"), str) and data["ref_text"].strip()
    ):
        raise ValueError("Cloning requires --ref-text/--ref-text-file, or --x-vector-only.")
    data["ref_audio"] = str(Path(audio).resolve())
    return data


def prepare(data, force=False, allow_stdin=True):
    data = dict(data)
    mode = data["mode"]
    for field in ("language", "size", "model", "instruct", "speaker", "voice", "save_voice"):
        if data.get(field) is not None and not isinstance(data[field], str):
            raise ValueError(f"{field} must be a string.")
    data["text"] = read_text(data, allow_stdin)
    data["language"] = canonical(data.get("language", "Auto"), LANGUAGES, "language")
    data["checkpoint"] = checkpoint(mode, data.get("size", "1.7B"), data.get("model"))
    output = Path(data.get("output", "output.wav")).expanduser().resolve()
    if output.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension.")
    if output.exists() and (not force or not output.is_file()):
        raise ValueError(f"Output exists: {output}. Use --force to replace a file.")
    data["output"] = output
    for field in ("temperature", "top_p", "repetition_penalty"):
        value = data.get(field)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or (field == "top_p" and value > 1)
        ):
            raise ValueError(f"{field} must be positive; top_p must also be ≤ 1.")
    for field in ("top_k", "max_new_tokens", "seed"):
        value = data.get(field)
        minimum = 1 if field == "max_new_tokens" else 0
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{field} must be an integer ≥ {minimum}.")
    if mode == "custom":
        data["speaker"] = canonical(data.get("speaker", "Ryan"), SPEAKERS, "speaker")
        if data.get("instruct") and "0.6B" in data["checkpoint"]:
            raise ValueError(
                "CustomVoice 0.6B ignores instructions. Use --size 1.7B for style control."
            )
    if mode == "design" and not (
        isinstance(data.get("instruct"), str) and data["instruct"].strip()
    ):
        raise ValueError("Voice design requires a non-empty --instruct description.")
    if mode == "clone":
        reference(data)
        if output == Path(data["ref_audio"]):
            raise ValueError("Output must differ from reference audio.")
    if data.get("save_voice"):
        if profiles.profile_path(data["save_voice"]).exists():
            raise ValueError(f"Voice profile {data['save_voice']!r} already exists.")
    return data


BATCH_COMMON = {
    "text",
    "text_file",
    "output",
    "language",
    "size",
    "model",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "repetition_penalty",
    "mode",
}
BATCH_MODE = {
    "design": {"instruct", "save_voice"},
    "custom": {"instruct", "speaker"},
    "clone": {"ref_audio", "ref_text", "ref_text_file", "x_vector_only", "voice", "save_voice"},
}


def batch_requests(path, force):
    requests, outputs, names = [], set(), set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get("mode") not in MODELS:
                raise ValueError("Each job needs mode: custom, design, or clone.")
            unknown = set(data) - BATCH_COMMON - BATCH_MODE[data["mode"]]
            if unknown:
                raise ValueError(f"Unknown fields: {', '.join(sorted(unknown))}")
            if not data.get("output"):
                raise ValueError("Each batch job needs an output path.")
            for key in ("text_file", "ref_audio", "ref_text_file", "output"):
                if data.get(key):
                    if data[key] == "-":
                        raise ValueError("Batch jobs cannot read stdin.")
                    data[key] = str((path.parent / data[key]).resolve())
            if data.get("model") and (path.parent / data["model"]).is_dir():
                data["model"] = str((path.parent / data["model"]).resolve())
            request = prepare(data, force, allow_stdin=False)
            if request["output"] in outputs:
                raise ValueError("Batch output paths must be unique.")
            name = request.get("save_voice")
            if name and name in names:
                raise ValueError("Batch profile names must be unique.")
            outputs.add(request["output"])
            names.add(name)
            requests.append(request)
        except (ValueError, TypeError, OSError, KeyError, AttributeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not requests:
        raise ValueError("Batch manifest has no jobs.")
    references = {Path(r["ref_audio"]) for r in requests if r.get("ref_audio")}
    if outputs & references:
        raise ValueError("Batch outputs must not overwrite any reference audio.")
    return requests


def generate(requests, args):
    engine = runtime.Runtime(args)
    for index, request in enumerate(requests, 1):
        print(f"[{index}/{len(requests)}] Generating {request['mode']}…", file=sys.stderr)
        audio, rate = engine.generate(request)
        runtime.write_audio(request["output"], audio, rate, args.force)
        if request.get("save_voice"):
            is_clone = request["mode"] == "clone"
            profiles.save(
                request["save_voice"],
                request["ref_audio"] if is_clone else request["output"],
                request.get("ref_text") if is_clone else request["text"],
                request.get("x_vector_only", False) if is_clone else False,
            )
        print(request["output"])
        if args.play:
            runtime.play(request["output"])


def doctor():
    print(f"Python: {sys.version.split()[0]}\nProfiles: {profiles.profile_root()}")
    for package in ("qwen-tts", "torch", "soundfile", "huggingface-hub"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: missing")
    try:
        import torch

        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("Install runtime dependencies: pip install -e .")


def dispatch(args):
    command = args.command
    if command in MODELS or command == "batch":
        if args.device not in ("auto", "cpu", "mps", "cuda") and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()
        ):
            raise ValueError("Device must be auto, cpu, mps, cuda, or cuda:N.")
        if command == "batch":
            requests = batch_requests(args.manifest.resolve(), args.force)
        else:
            requests = [prepare({**vars(args), "mode": command}, args.force)]
        runtime.configure_environment(args.offline, args.cache_dir)
        generate(requests, args)
    elif command == "download":
        runtime.configure_environment(args.offline, args.cache_dir)
        print(runtime.download(checkpoint(args.mode, args.size, args.model), args.offline))
    elif command == "models":
        for mode, sizes in MODELS.items():
            for size, model in sizes.items():
                print(f"{mode:8} {size:4} {model}")
    elif command == "speakers":
        for name, description in SPEAKERS.items():
            print(f"{name:10} {description}")
    elif command == "languages":
        print("\n".join(LANGUAGES))
    elif command == "doctor":
        doctor()
    elif command == "voices":
        if args.action == "list":
            print(
                "\n".join(profiles.names()) or "No saved voices. Use 'voices add' or --save-voice."
            )
        elif args.action == "show":
            print(json.dumps(profiles.load(args.name), indent=2))
        elif args.action == "delete":
            path = profiles.profile_path(args.name)
            if not path.is_dir():
                raise ValueError(f"Voice profile {args.name!r} does not exist.")
            shutil.rmtree(path)
            print(f"Deleted voice {args.name}")
        elif args.action == "add":
            data = reference(vars(args).copy())
            print(
                profiles.save(
                    args.name, data["ref_audio"], data.get("ref_text"), args.x_vector_only
                )
            )


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except ImportError as error:
        print(
            f"Missing runtime dependency: {error}. Install with: pip install -e .", file=sys.stderr
        )
        raise SystemExit(1) from None
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        if getattr(args, "offline", False):
            print(
                "Offline mode: download this checkpoint first with 'qwen3-tts download MODE'.",
                file=sys.stderr,
            )
        raise SystemExit(1) from None
