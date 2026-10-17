# Batch narration

Use `qwen3-tts batch` for multiple outputs or sections of a long script. It runs
jobs sequentially, producing one WAV per job; it does not concatenate them.

## Manifest format

Write UTF-8 JSONL: one JSON object per nonblank line. Use a JSON serializer for
text with quotes, backslashes, or line breaks. Each job requires `mode`, `output`,
and either `text` or `text_file`. Paths are relative to the manifest directory,
not the terminal's working directory. If the user specifies no output location,
resolve `./qwen3-tts-output/` from the agent's current working directory and use
absolute output paths in the manifest, or keep the manifest in that working
directory as the examples below assume. This avoids creating a nested output
folder when the manifest itself lives elsewhere. Use the user's specified
location instead when provided. Apply pronunciation-friendly spellings to
agent-written `text` or `text_file` content just as for single generations.

A manifest named `jobs.jsonl` alongside the referenced files could contain:

```jsonl
{"mode":"custom","text":"Welcome to the tutorial.","speaker":"Ryan","language":"English","output":"./qwen3-tts-output/welcome.wav"}
{"mode":"design","text":"Here is a short sample of our narrator.","instruct":"A warm adult male narrator with precise diction and a steady conversational pace.","language":"English","output":"./qwen3-tts-output/reference.wav","save_voice":"new-narrator"}
{"mode":"clone","ref_audio":"speaker.wav","ref_text_file":"speaker-transcript.txt","text_file":"chapter-01.txt","language":"English","output":"./qwen3-tts-output/chapter-01.wav"}
```

To reuse an **already existing** profile in a separate manifest:

```jsonl
{"mode":"clone","voice":"tutorial-narrator","language":"French","text_file":"chapter-01.txt","output":"./qwen3-tts-output/chapter-01.wav"}
{"mode":"clone","voice":"tutorial-narrator","language":"French","text_file":"chapter-02.txt","output":"./qwen3-tts-output/chapter-02.wav"}
```

```sh
qwen3-tts batch jobs.jsonl
qwen3-tts batch jobs.jsonl --offline --device cpu
```

## Fields

| Scope | JSON fields |
| --- | --- |
| All jobs | `mode`, `text` or `text_file`, `output`, `language`, `size`, `model`, `seed`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `repetition_penalty` |
| CustomVoice | `speaker`, optional `instruct` (1.7B only) |
| VoiceDesign | Required `instruct`, optional `save_voice` |
| VoiceCloning | `voice` OR `ref_audio` plus `ref_text`/`ref_text_file` or `x_vector_only`; optional `save_voice` |

Use underscores in JSON field names. `x_vector_only` is a JSON boolean (`true` or
`false`), not a quoted string. Use only the fields for the job's mode.

Runtime options are batch CLI flags, **not job fields**: `--offline`, `--cache-dir`,
`--device`, `--dtype`, `--attention`, `--force`, and `--play`. There is no batch-wide
`--size`, `--language`, or `--seed`; put those in each job when needed.

## Execution constraints

- All jobs are prevalidated before synthesis. A voice saved by an earlier job in
  the same batch is unavailable during prevalidation. Finish voice creation in a
  separate command before batching speech with that profile.
- Output paths and newly saved profile names must be unique. Outputs must not
  overwrite reference audio. Stdin (`text_file: "-"`) is unavailable in jobs.
- Group adjacent jobs by checkpoint to reuse the loaded model. Switching modes
  or sizes releases the previous model and loads the next one.
- Runtime failure stops the batch. Previously completed files remain. Inspect
  them and rerun only unfinished jobs using a revised manifest; do not blindly
  force-overwrite successful outputs.
