# Installation, profiles, offline use, and recovery

Read installed `--help` before relying on options in a newer/different version.
The installed CLI is independent of the source checkout and can run from any
working directory. Copying this skill does not install the executable or weights.

## Installation when needed

On another Mac, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
if needed, then install the CLI from GitHub:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv tool install --python 3.12 git+https://github.com/MaximeBF2000/qwen-3-tts-cli.git
uv tool update-shell
```

Open a new terminal, then run `qwen3-tts --help`. No source checkout or virtual
environment activation is needed. If the agent retains an old PATH, use the
installed executable directly, commonly `~/.local/bin/qwen3-tts`.

Use `qwen3-tts doctor` for package versions, profile location, and device status.
Sandbox restrictions may hide MPS even on a capable Mac; a restricted diagnostic
alone does not establish the machine's hardware capabilities. Follow the agent's
normal execution permissions for any broader device, cache, or profile access.

## Local profile management

```sh
qwen3-tts voices add my-narrator --ref-audio reference.wav --ref-text-file transcript.txt
qwen3-tts voices add embedding-only --ref-audio reference.wav --x-vector-only
qwen3-tts voices list
qwen3-tts voices show my-narrator
qwen3-tts voices delete my-narrator
```

Use deletion only when requested or already authorized. Profiles reside in
`~/.local/share/qwen3-tts/voices/NAME/` by default. `QWEN3_TTS_HOME` overrides the
root. A profile copies its reference audio and stores its transcript and cloning
mode in JSON. It survives moving the original recording; features are recomputed
on use. `voices add` does not load a model or generate new audio.

## Models and offline operation

```sh
qwen3-tts models
qwen3-tts download clone
qwen3-tts download custom --size 0.6B --cache-dir ./model-cache
qwen3-tts custom --size 0.6B --cache-dir ./model-cache --offline \
  --speaker Ryan --text "Entirely local." -o ./qwen3-tts-output/local.wav
```

Select the same mode, size, and cache directory for download and generation.
Designing a reference does not cache the separate Base checkpoint needed to
clone it. `--model /path/to/checkpoint` accepts a complete compatible local model,
including `speech_tokenizer/`, or an alternate Hugging Face ID. Choose a model
matching the command's mode.

`--offline` disables Hub access and requires complete local files. `HF_HOME`,
`HF_HUB_CACHE`, and `HF_HUB_OFFLINE` are respected; `--cache-dir` overrides the Hub
cache directory. If offline files are missing, report which download is needed
without silently enabling network access.

## Targeted recovery

- **Command not found:** inspect PATH/installation before reinstalling. A tool
  installed in another Python environment need not be importable by system Python.
- **Missing dependencies or conflicts:** use the tool's isolated environment;
  avoid changing unrelated system/project Python packages.
- **MPS failure:** auto already retries CPU once. If it still fails, inspect the
  final error. `--device cpu` explicitly selects compatibility mode. Clone/custom
  offer `--size 0.6B` for memory pressure; design does not.
- **CUDA/FlashAttention:** default auto prefers MPS/CPU, not CUDA. CUDA requires
  explicit selection and compatible PyTorch. Optional `flash_attention_2` needs
  its separately installed package, a compatible GPU, and float16/bfloat16.
  Do not install it merely because upstream prints a speed warning on a Mac.
- **Output/profile exists:** inspect the existing artifact. Reuse a profile,
  select a new name, or replace a WAV only when the task calls for replacement.
- **Long narration truncates/repeats:** split the script into logical shorter
  utterances. `--max-new-tokens` limits codec tokens, not seconds. Advanced
  sampling controls include `--temperature`, `--top-p`, `--top-k`, and
  `--repetition-penalty`; retain defaults without evidence they need changing.
- **Playback fails:** the WAV may already be complete. macOS uses `afplay`;
  other platforms require the package's `playback` extra, PortAudio where needed,
  and an accessible audio device. Return the file instead of resynthesizing it.

`--seed` improves repeatability within an environment, but does not guarantee
identical output across hardware. Output is 16-bit PCM WAV at the model's sample
rate. Exit status is 0 on success, 1 on runtime/input failure, 2 on argument
parsing failure, and 130 on interruption.
