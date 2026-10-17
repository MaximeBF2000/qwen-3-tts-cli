---
name: qwen3-tts
description: Generate local speech and voiceovers with the installed qwen3-tts CLI. Use for text-to-speech, tutorial narration, natural-language voice design, voice cloning from recordings, reusable voice profiles, and batch audio generation with Qwen3-TTS.
---

# Qwen3-TTS

Use the installed `qwen3-tts` command to produce local WAV audio. This skill works
from any project directory; it does not require the CLI source checkout, Python
imports, or a particular agent's tools.

## Establish the workflow

- Check `qwen3-tts --version` and consult the relevant subcommand's `--help` when
  needed. Treat installed help as authoritative if it differs from this skill.
- If the executable is missing, check the current PATH and the user's installation.
  A uv tool install normally exposes it in `~/.local/bin`. Use the discovered
  executable path if necessary; do not reinstall a working tool into the project.
  If genuinely uninstalled, explain the prerequisite. Installation instructions
  and recovery options are in [references/operations.md](references/operations.md).
- Infer language and delivery from the request and context. Use the user's
  specified output location; otherwise store all produced audio in
  `./qwen3-tts-output/` relative to the working directory.
  Ask only for missing information that materially affects the result, such as
  which recording to clone. Example French commands below do not set a default
  language for other users.
- Select the mode using the table below. When reusing a voice, inspect existing
  names before generating. Execute the requested workflow; do not stop after
  writing a command unless the user asked only for instructions.

| User intent | Command | Voice input |
| --- | --- | --- |
| Use a built-in speaker | `custom` | `--speaker` from `speakers` |
| Invent a voice from a description | `design` | Required `--instruct` |
| Clone a local recording | `clone` | `--ref-audio` and reference transcript |
| Reuse any saved voice, including a designed one | `clone` | `--voice NAME` |
| Generate multiple utterances | `batch` | JSONL manifest; read [references/batch.md](references/batch.md) |

## Discover speakers and saved voices

```sh
qwen3-tts speakers
qwen3-tts voices list
qwen3-tts voices show tutorial-narrator
qwen3-tts languages
```

`speakers` lists built-in CustomVoice speakers. `voices list` lists saved local
profiles: designed and cloned voices share one namespace and their origin is not
recorded. Do not invent separate design/clone listing commands or infer origin
from the names. A saved profile is supplied with `--voice`, not `--speaker`.

## Produce speech

Always pass an explicit `.wav` output path. When the user specifies no location,
put every generated WAV—including previews, reference samples, variants, and
batch chunks—in `./qwen3-tts-output/` under the current working directory. Use
the location the user mentions instead when provided. The CLI creates output
parent directories. This is an agent convention; the CLI itself otherwise
defaults to `output.wav`. Saved profile copies still use the normal voice library.

Keep runtime flags after the subcommand name. For long or shell-sensitive text,
write a UTF-8 file with a file-writing tool or a literal heredoc and use
`--text-file`; avoid interpolating narration into shell code. `--text` and
`--text-file` are mutually exclusive. Explicit input prevents waiting on stdin.

### Write for pronunciation

When you compose the narration yourself, write the TTS input for how it should
sound. Use phonetic respelling in the target language for uncommon words, brand
names, acronyms, or names likely to be mispronounced. For example, if the brand
`Jev` is pronounced “Djev”, send `Djev` in the spoken text:

```sh
qwen3-tts custom --speaker Ryan --language French \
  --text "Aujourd'hui, nous allons découvrir Djev." \
  -o ./qwen3-tts-output/jev-introduction.wav
```

Keep ordinary words readable; use plain pronunciation-friendly spelling rather
than IPA symbols, SSML, or bracketed directions that could be read aloud. Use the
user's stated pronunciation when available; clarify a consequential ambiguous
brand/name rather than guessing. Preserve the intended meaning. If also producing
an on-screen script or captions, keep the correct written brand name (`Jev`) there
and the pronunciation spelling (`Djev`) in the TTS input.

For user-supplied text that must be read verbatim, preserve it unless pronunciation
edits are requested. Do not phoneticize an existing recording's reference
transcript: it must remain an accurate transcript of that recording.

### Built-in speaker

```sh
qwen3-tts custom --speaker Ryan --language French \
  --text "Bonjour ! Nous allons avancer étape par étape." \
  --instruct "Speak clearly, warmly, and at a measured conversational pace." \
  -o ./qwen3-tts-output/introduction.wav
```

Choose a speaker from the installed catalog. All published speakers can speak
French and the other supported languages; native language affects voice quality.
`--instruct` controls delivery on CustomVoice 1.7B. CustomVoice 0.6B does not
support instructions.

### Design and save a reusable voice

Translate the user's purpose into a coherent voice description: language/accent,
perceived age, pitch/timbre, clarity, pace, expression, and delivery. For tutorial
narration, favor clear diction, conversational authority, purposeful pauses, and
restrained enthusiasm. Adapt those choices to the user rather than applying one
fixed persona to every task.

Generate a short representative passage in the target language. Around 5–15
seconds of speech is a practical starting point for a reusable reference, not a
CLI duration constraint. Use the requested profile name exactly if valid; names
must start with a letter or digit, contain only letters, digits, `_` or `-`, and
be 1–64 characters long.

```sh
qwen3-tts design --language French \
  --text "Dans ce tutoriel, nous allons avancer ensemble. Je vous montrerai chaque étape et vous expliquerai pourquoi elle compte. À la fin, vous saurez refaire la méthode par vous-même." \
  --instruct "Native French adult male voice, medium-low pitch, warm clear timbre. An approachable, knowledgeable mentor. Precise diction, moderate conversational pace, short purposeful pauses, subtle narrative expression, and calm confidence." \
  --seed 42 -o ./qwen3-tts-output/narrator-reference.wav --save-voice tutorial-narrator
```

`--save-voice` copies the generated audio and its exact text into a local profile.
Use that profile through `clone` for subsequent lines; rerunning `design` is not
how to preserve a consistent character across a series.

```sh
qwen3-tts clone --voice tutorial-narrator --language French \
  --text-file narration.txt -o ./qwen3-tts-output/narration.wav
```

Do not assume the example profile exists: create it or select an existing one.

### Clone an existing recording

Use an existing local audio file with one clear speaker and minimal noise,
music, or silence. Qwen supports short references around three seconds; a clean
5–15 second passage is a useful starting point. The CLI imposes no duration cap.
Only use recordings the user has permission to clone.

```sh
qwen3-tts clone --ref-audio reference.wav \
  --ref-text-file reference-transcript.txt --language French \
  --text-file narration.txt -o ./qwen3-tts-output/cloned.wav --save-voice my-narrator
```

The reference transcript must match the words actually spoken in the recording,
in that recording's language. `--language` specifies the output language.
Do not substitute the target narration for the reference transcript or invent
missing reference words. If no transcript is available, explain that
`--x-vector-only` skips it at reduced fidelity. The CLI does not transcribe audio.

For `clone --voice NAME`, omit `--ref-audio`, `--ref-text`, `--ref-text-file`, and
`--x-vector-only`: the profile already supplies these settings. Clone has no
`--instruct` or `--speaker` option. A profile stores reference audio and metadata,
not a newly trained or fine-tuned model.

## Execution defaults and limits

- Keep automatic device selection and SDPA unless the task requires otherwise.
  Auto selects MPS on Apple Silicon when available and otherwise CPU; it retries
  CPU once after an MPS runtime failure. CUDA is explicit with `--device cuda:0`.
- Each mode uses a separate checkpoint, defaulting to 1.7B. Design only has 1.7B;
  custom and clone also offer `--size 0.6B`. First use may download several GB.
  Local synthesis does not mean the weights are already cached. Respect a user's
  offline constraint using `--offline`; do not silently retry online.
- Existing WAVs are protected unless `--force` is used. Saved profile names cannot
  be overwritten, even with `--force`. Choose a new name/path when appropriate;
  only delete profiles or replace files when that matches the user's request.
- `--save-voice` is available on design and clone. To save an existing reference
  without synthesis, use `voices add`; see the operations reference.
- A text file is one utterance, not automatic chunking. Split long scripts into
  logical paragraphs/sections and batch them. Keep one saved voice across chunks.
  The CLI does not merge files, normalize loudness, export MP3, or stream audio.
- Playback is optional with `--play`. macOS uses `afplay`; other systems need the
  playback extra and a working audio device. Playback failure can occur after
  successful WAV creation, so check the file before regenerating.

## Verify and deliver

Wait for completion while reporting meaningful progress during downloads or
inference. A success log prints the output path to stdout; progress/errors go to
stderr. Inspect an error rather than repeatedly launching the same generation.
For setup, cache, hardware, or playback issues, read
[references/operations.md](references/operations.md).

Verify the WAV exists and is nonempty. If audio inspection is available, check
that it decodes, has nonzero duration, and is not silent; listen when the task
requires judging pronunciation or delivery. Do not claim to have listened based
only on file metadata. Verify a newly saved profile with `voices show NAME`.

Return a link or playable preview using the host agent's supported format, the
saved profile name if applicable, and a short reuse command when useful. Distinguish
successful file generation from any unverified subjective audio quality. Further
variants should serve the user's request, rather than trigger an open-ended
search for a perfect voice.
