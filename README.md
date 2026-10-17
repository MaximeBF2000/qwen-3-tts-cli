# Qwen3-TTS CLI

A local-first Python command line tool for **VoiceDesign**, **VoiceCloning**, and
**CustomVoice**, using the official [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
Python package. Audio is synthesized on your computer. No hosted inference service or API key.

## Install on a Mac

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then install the CLI directly from GitHub:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv tool install --python 3.12 git+https://github.com/MaximeBF2000/qwen-3-tts-cli.git
uv tool update-shell
```

Open a new terminal. You can now use `qwen3-tts` from any directory:

```sh
qwen3-tts doctor
qwen3-tts --help
```

To update after new changes are pushed:

```sh
uv tool upgrade qwen3-tts-cli
```

On Apple Silicon, the CLI automatically uses MPS when available and falls back to
CPU. Required models download on first use.

## Quick start

```sh
# Preset speaker (CustomVoice)
qwen3-tts custom --text 'Hello from my Mac!' --speaker Ryan -o hello.wav --play

# Describe a new voice (VoiceDesign)
qwen3-tts design --text 'Welcome to the show.' \
  --instruct 'A warm, low-pitched narrator, speaking calmly and clearly.' \
  --language English -o narrator.wav --save-voice narrator

# Clone a local recording using its exact transcript (Base model)
qwen3-tts clone --ref-audio reference.wav \
  --ref-text 'The exact words spoken in the reference recording.' \
  --text 'Here is something new in the same voice.' -o cloned.wav

# Reuse the designed voice through the Base cloning model
qwen3-tts clone --voice narrator --text 'Our story continues.' -o next.wav
```

Each mode selects its own **1.7B** checkpoint by default. Clone and CustomVoice
also support `--size 0.6B`. VoiceDesign only has a 1.7B checkpoint. CustomVoice
0.6B does not support style instructions; use 1.7B with `--instruct`.
`--model /path/to/checkpoint` overrides checkpoint selection and can point to a
compatible fine-tuned model directory. The directory must contain the complete
checkpoint, including its `speech_tokenizer/` files. Model overrides must match
the command's mode; upstream validates compatibility.

## Text, language, and generation settings

```sh
qwen3-tts custom --text-file script.txt --speaker Aiden -o script.wav
cat script.txt | qwen3-tts custom --speaker Ryan -o script.wav
qwen3-tts custom --text-file - --speaker Ryan -o script.wav < script.txt
qwen3-tts custom --text 'Bonjour !' --language French --speaker Serena \
  --instruct 'Speak gently and with enthusiasm.' -o bonjour.wav
qwen3-tts custom --text 'A repeatable experiment.' --seed 42 \
  --temperature 0.8 --top-p 0.95 --top-k 50 --max-new-tokens 2048 \
  --repetition-penalty 1.05 -o experiment.wav
```

`--text`, `--text-file`, and stdin provide one utterance per command; a text file
is read as a whole, with no automatic long-document chunking. Split long content
into batch jobs to control duration and avoid hitting generation limits.
`--language Auto` is the default. Use `qwen3-tts languages` and `qwen3-tts speakers`
to discover options. Language and speaker names are case-insensitive. Use a
speaker's native language for best results. Seeds improve repeatability within
the same environment; they do not guarantee identical audio across devices.

Output is 16-bit PCM WAV at the model's returned sample rate. Existing output
files are protected unless `--force` is set. Output paths go to stdout; progress
and errors go to stderr. Exit codes: 0 success, 1 runtime/input failure, 2 argument
parsing failure, 130 interruption. Runtime options go **after the command name**.

## Saved voices

```sh
qwen3-tts voices add my-voice --ref-audio reference.wav --ref-text-file transcript.txt
qwen3-tts voices list
qwen3-tts voices show my-voice
qwen3-tts clone --voice my-voice --text 'Good morning.' -o morning.wav
qwen3-tts voices delete my-voice

# Reference transcript unavailable: embedding-only cloning, with reduced fidelity
qwen3-tts clone --ref-audio reference.wav --x-vector-only \
  --text 'Hello again.' -o greeting.wav --save-voice quick-voice
```

Profiles copy reference audio and its transcript into
`~/.local/share/qwen3-tts/voices/NAME/`. Override the root with `QWEN3_TTS_HOME`.
They are ordinary audio + JSON, portable between machines, and remain available
if the original recording is moved. Reference features are recomputed when used;
profiles do not store model tensors. Existing profile names are protected.
`--save-voice` on design stores the generated audio with its text, enabling the
upstream design-then-clone workflow. On clone it preserves the original reference.
Only clone recordings you have permission to use.

## Downloads and offline use

First generation downloads the required model from Hugging Face if missing.
This is the only network requirement; generation uses local files. Checkpoints
are large, and each mode needs a separate one. No weights are bundled.

```sh
qwen3-tts models
qwen3-tts download custom --size 0.6B
qwen3-tts custom --size 0.6B --offline --text 'Entirely local.' -o local.wav

# Put downloads in a particular cache and reuse it offline
qwen3-tts download design --cache-dir ./model-cache
qwen3-tts design --cache-dir ./model-cache --offline --text 'Hello.' \
  --instruct 'A relaxed storyteller.' -o story.wav

# Use a complete checkpoint directory directly
qwen3-tts clone --model /path/to/Base --offline --voice my-voice \
  --text 'No network needed.' -o offline.wav
```

`--offline` enables Hugging Face/Transformers offline settings before importing
them, resolves only local snapshots, and fails with a useful error if files are
missing. Reference audio accepts local files only. `HF_HUB_CACHE`, `HF_HOME`, and
`HF_HUB_OFFLINE` are also respected; `--cache-dir` overrides the Hub cache path.
The checkpoint snapshot includes the bundled speech tokenizer.

## Batch generation

Create a UTF-8 JSONL file with one job per line. See [examples/batch.jsonl](examples/batch.jsonl).
All jobs are validated before loading a model or writing audio. Each needs `mode`,
text (or `text_file`), and `output`. Paths are relative to the manifest directory.

```jsonl
{"mode":"custom","text":"First line.","speaker":"Ryan","output":"out/first.wav"}
{"mode":"design","text":"Second line.","instruct":"A cheerful storyteller.","output":"out/second.wav"}
```

```sh
qwen3-tts batch examples/batch.jsonl --offline
qwen3-tts batch jobs.jsonl --play --device cpu
```

Jobs may use the corresponding generation command's fields, replacing hyphens
with underscores (for example `ref_audio`, `save_voice`, `max_new_tokens`).
Runtime options (`offline`, `device`, `dtype`, `attention`, `cache_dir`, `force`,
`play`) apply to the whole batch via CLI flags. Stdin is unavailable inside jobs.
Saved voices must exist before a batch starts. Output paths and new profile names
must be unique. Jobs run sequentially and reuse the loaded model when adjacent
jobs select the same checkpoint; switching checkpoints releases the previous one
to limit memory usage. Group similar jobs together for speed. The batch stops on
the first runtime failure; completed files remain available.

## Playback and troubleshooting

`--play` uses macOS's built-in `afplay`. Playback occurs after writing the WAV,
so the file remains available if the audio device fails.

- **MPS error or memory pressure:** use `--device cpu`, or `--size 0.6B` on clone/custom.
  Auto mode retries CPU once, which still requires enough system memory.
- **Missing package or dependency conflict:** reinstall with `uv tool install --force
  --python 3.12 git+https://github.com/MaximeBF2000/qwen-3-tts-cli.git`, then run
  `qwen3-tts doctor`.
- **Offline cache miss:** run `download` with the same mode, size, and cache directory online.
- **Truncated or repetitive speech:** split long text into shorter jobs and adjust
  `--max-new-tokens` or `--repetition-penalty`.
- **Advanced CUDA setup:** install a compatible PyTorch CUDA build separately.
  FlashAttention 2 is optional and requires a compatible CUDA GPU, the separately
  installed `flash-attn` package, and `--dtype float16` or `bfloat16`.

The CLI writes complete WAV files; it does not expose live streaming. The
upstream Python `generate_*` APIs return completed waveforms even though the
model family supports streaming in other integrations.

Built against the official [Python API](https://github.com/QwenLM/Qwen3-TTS#python-package-usage)
and [model loader](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py).
