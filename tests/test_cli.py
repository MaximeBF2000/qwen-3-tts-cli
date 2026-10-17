import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qwen3_tts_cli import cli, profiles, runtime


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.env = patch.dict(os.environ, {"QWEN3_TTS_HOME": str(self.root / "profiles")})
        self.env.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.env.stop)

    def job(self, mode="custom", **kwargs):
        return {"mode": mode, "text": "Hello", "output": str(self.root / "out.wav"), **kwargs}

    def test_validation_before_model_loading(self):
        invalid = [
            self.job(text=" "),
            self.job(language="Klingon"),
            self.job(speaker="missing"),
            self.job(temperature=float("nan")),
            self.job(top_p=1.1),
            self.job(max_new_tokens=0),
            self.job("design", instruct="warm", size="0.6B"),
            self.job(size="0.6B", instruct="angry"),
            self.job("clone"),
        ]
        for job in invalid:
            with self.subTest(job=job), self.assertRaises(ValueError):
                cli.prepare(job)

    def test_case_insensitive_language_and_speaker(self):
        job = cli.prepare(self.job(language="french", speaker="ryan"))
        self.assertEqual((job["language"], job["speaker"]), ("French", "Ryan"))

    def test_file_stdin_and_unicode(self):
        path = self.root / "text.txt"
        path.write_text("Bonjour à tous", encoding="utf-8")
        self.assertEqual(cli.read_text({"text_file": str(path)}), "Bonjour à tous")
        with patch("sys.stdin", io.StringIO("Hello from stdin")):
            self.assertEqual(cli.read_text({}), "Hello from stdin")
        with self.assertRaises(ValueError):
            cli.read_text({"text": "a", "text_file": str(path)})

    def test_profiles_copy_audio_and_roundtrip_transcript(self):
        audio = self.root / "ref.wav"
        audio.write_bytes(b"fixture")
        profiles.save("voice", audio, "Reference words")
        audio.unlink()
        self.assertEqual(profiles.names(), ["voice"])
        job = cli.prepare(self.job("clone", voice="voice"))
        self.assertEqual(job["ref_text"], "Reference words")
        self.assertEqual(Path(job["ref_audio"]).read_bytes(), b"fixture")
        with self.assertRaises(ValueError):
            profiles.save("voice", Path(job["ref_audio"]), "words")
        with self.assertRaises(ValueError):
            profiles.profile_path("../escape")

    def test_clone_reference_rules(self):
        audio = self.root / "ref.wav"
        audio.touch()
        cli.prepare(self.job("clone", ref_audio=str(audio), x_vector_only=True))
        with self.assertRaises(ValueError):
            cli.prepare(self.job("clone", ref_audio=str(audio)))
        with self.assertRaises(ValueError):
            cli.prepare(
                self.job("clone", ref_audio=str(audio), ref_text="hello", output=str(audio)),
                force=True,
            )

    def test_batch_relative_paths_and_preflight(self):
        manifest = self.root / "jobs.jsonl"
        manifest.write_text(json.dumps({"mode": "custom", "text": "a", "output": "a.wav"}))
        jobs = cli.batch_requests(manifest, False)
        self.assertEqual(jobs[0]["output"], self.root / "a.wav")
        manifest.write_text(
            "\n".join(
                json.dumps({"mode": "custom", "text": "a", "output": "a.wav"}) for _ in range(2)
            )
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            cli.batch_requests(manifest, False)
        manifest.write_text('{"mode":"custom","text_file":"-","output":"b.wav"}')
        with self.assertRaisesRegex(ValueError, "stdin"):
            cli.batch_requests(manifest, False)
        self.assertFalse((self.root / "a.wav").exists())

    def test_complete_command_writes_playable_wav_and_profile(self):
        import soundfile as sf
        import numpy as np

        engine = MagicMock()
        engine.generate.return_value = (np.zeros(2400, dtype=np.float32), 24000)
        output = self.root / "test.wav"
        with (
            patch.object(runtime, "Runtime", return_value=engine),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main(
                [
                    "design",
                    "--text",
                    "Hello",
                    "--instruct",
                    "Warm narrator",
                    "-o",
                    str(output),
                    "--save-voice",
                    "narrator",
                ]
            )
        info = sf.info(output)
        self.assertEqual((info.samplerate, info.frames, info.subtype), (24000, 2400, "PCM_16"))
        self.assertEqual(profiles.load("narrator")["ref_text"], "Hello")
        with self.assertRaises(FileExistsError):
            runtime.write_audio(output, [0.0], 24000)

    def test_failed_wav_write_preserves_existing_output(self):
        output = self.root / "existing.wav"
        output.write_bytes(b"original")
        with patch("soundfile.write", side_effect=RuntimeError("write failed")):
            with self.assertRaises(RuntimeError):
                runtime.write_audio(output, [0.0], 24000, force=True)
        self.assertEqual(output.read_bytes(), b"original")
        self.assertEqual(list(self.root.glob("*.wav")), [output])

    def test_batch_rejects_non_boolean_embedding_flag(self):
        audio = self.root / "ref.wav"
        audio.touch()
        with self.assertRaisesRegex(ValueError, "boolean"):
            cli.prepare(self.job("clone", ref_audio=str(audio), x_vector_only="false"))

    def test_help_does_not_import_runtime_dependencies(self):
        script = "from qwen3_tts_cli.cli import main; import sys\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'torch' not in sys.modules\nassert 'qwen_tts' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


class AdapterTests(unittest.TestCase):
    def args(self, **kwargs):
        return argparse.Namespace(
            device="auto", dtype="auto", attention="sdpa", offline=False, **kwargs
        )

    def fake_torch(self):
        torch = MagicMock()
        torch.backends.mps.is_available.return_value = True
        torch.float32 = "float32"
        return torch

    def test_dispatch_three_upstream_methods_and_model_reuse(self):
        torch = self.fake_torch()
        model = MagicMock()
        for method in ("generate_custom_voice", "generate_voice_design", "generate_voice_clone"):
            getattr(model, method).return_value = ([[0.1, 0.2]], 24000)
        factory = MagicMock()
        factory.from_pretrained.return_value = model
        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=factory)},
            ),
            patch.object(runtime, "resolve_model", return_value="/local/checkpoint"),
        ):
            engine = runtime.Runtime(self.args())
            for mode in ("custom", "design", "clone"):
                engine.generate(
                    {
                        "mode": mode,
                        "checkpoint": "test",
                        "text": "Hello",
                        "seed": 1,
                        "language": "English",
                        "speaker": "Ryan",
                        "instruct": "warm",
                        "ref_audio": "/ref.wav",
                        "ref_text": "words",
                        "temperature": 0.8,
                    }
                )
            factory.from_pretrained.assert_called_once_with(
                "/local/checkpoint", device_map="mps", dtype="float32", attn_implementation="sdpa"
            )
            model.generate_voice_clone.assert_called_once_with(
                text="Hello",
                language="English",
                temperature=0.8,
                ref_audio="/ref.wav",
                ref_text="words",
                x_vector_only_mode=False,
            )
            self.assertEqual(model.generate_custom_voice.call_args.kwargs["speaker"], "Ryan")
            self.assertEqual(model.generate_voice_design.call_args.kwargs["instruct"], "warm")

    def test_mps_failure_reloads_on_cpu(self):
        torch = self.fake_torch()
        model = MagicMock()
        model.generate_custom_voice.side_effect = [
            RuntimeError("MPS unsupported"),
            ([[0.1]], 24000),
        ]
        factory = MagicMock()
        factory.from_pretrained.return_value = model
        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=factory)},
            ),
            patch.object(runtime, "resolve_model", return_value="/local"),
        ):
            engine = runtime.Runtime(self.args())
            engine.generate(
                {
                    "mode": "custom",
                    "checkpoint": "test",
                    "text": "Hi",
                    "language": "Auto",
                    "speaker": "Ryan",
                }
            )
            self.assertEqual(factory.from_pretrained.call_args.kwargs["device_map"], "cpu")
            self.assertEqual(factory.from_pretrained.call_count, 2)

    def test_offline_configuration_precedes_hub_load(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime.configure_environment(True, "/tmp/test-cache")
            hub = MagicMock()
            hub.snapshot_download.return_value = "/cached/snapshot"
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(runtime.resolve_model("Qwen/test", True), "/cached/snapshot")
            hub.snapshot_download.assert_called_once_with(
                repo_id="Qwen/test", local_files_only=True
            )
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(Path("/tmp/test-cache").resolve()))


if __name__ == "__main__":
    unittest.main()
