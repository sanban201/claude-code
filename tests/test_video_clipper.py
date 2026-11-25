import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from video_clipper import (
    ClipperError,
    ClipMoment,
    Word,
    _ollama_chat,
    check_ollama,
    deduplicate_moments,
    ensure_runtime,
    make_srt,
    render_clip,
    transcript_chunks,
    translate_captions,
    validate_moments,
    validate_video_url,
)


class VideoClipperTests(unittest.TestCase):
    def test_accepts_public_http_urls_and_rejects_local_targets(self):
        self.assertEqual(
            validate_video_url(" https://example.com/watch?v=123 "),
            "https://example.com/watch?v=123",
        )
        for value in ("file:///tmp/video.mp4", "http://localhost/video", "http://127.0.0.1/"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_video_url(value)

    def test_requires_ffmpeg_subtitle_filter(self):
        with patch("video_clipper.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
            "video_clipper.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="subtitles", stderr=""),
        ):
            self.assertEqual(ensure_runtime(), "/usr/bin/ffmpeg")

        with patch("video_clipper.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
            "video_clipper.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="no libass", stderr=""),
        ), self.assertRaises(ClipperError):
            ensure_runtime()

    def test_ollama_requests_use_configured_endpoint_and_bearer_key(self):
        def fake_tags(request, timeout):
            self.assertEqual(request.full_url, "https://ollama.example/api/tags")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
            self.assertEqual(timeout, 5)
            return io.BytesIO(b'{"models":[{"name":"llama3.2:latest"}]}')

        with patch("video_clipper.urllib.request.urlopen", side_effect=fake_tags):
            check_ollama("https://ollama.example/", api_key="secret-token")

        def fake_chat(request, timeout):
            self.assertEqual(request.full_url, "https://ollama.example/api/chat")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
            self.assertEqual(json.loads(request.data)["model"], "llama3.2")
            return io.BytesIO(b'{"message":{"content":"{}"}}')

        with patch("video_clipper.urllib.request.urlopen", side_effect=fake_chat):
            self.assertEqual(
                _ollama_chat(
                    "prompt",
                    "https://ollama.example/",
                    "llama3.2",
                    api_key="secret-token",
                ),
                "{}",
            )

    def test_transcript_chunks_keep_context_around_non_overlapping_cores(self):
        words = [Word(float(index), float(index + 1), "word") for index in range(600)]
        chunks = transcript_chunks(words, 600)

        self.assertEqual(
            [(chunk.core_start, chunk.core_end) for chunk in chunks],
            [(0.0, 240.0), (240.0, 480.0), (480.0, 600.0)],
        )
        self.assertEqual((chunks[1].context_start, chunks[1].context_end), (180.0, 540.0))

    def test_rejects_invalid_clip_durations_and_keeps_all_valid_moments(self):
        proposals = [
            {"start": 10, "end": 40, "reason": "valid"},
            {"start": 20, "end": 49, "reason": "too short"},
            {"start": 75, "end": 105, "reason": "past duration"},
            {"start": 250, "end": 290, "reason": "outside chunk core"},
        ]

        moments, rejected = validate_moments(proposals, 100, 0, 100)

        self.assertEqual([(item.start, item.end) for item in moments], [(10, 40)])
        self.assertEqual(rejected, 3)

    def test_deduplicates_overlapping_proposals_without_a_clip_count_limit(self):
        moments = [
            ClipMoment(0, 40, "one"),
            ClipMoment(2, 42, "duplicate"),
            ClipMoment(50, 90, "two"),
            ClipMoment(100, 145, "three"),
            ClipMoment(150, 195, "four"),
        ]

        self.assertEqual(
            [item.reason for item in deduplicate_moments(moments)],
            ["one", "two", "three", "four"],
        )

    def test_srt_uses_clip_relative_word_timestamps_and_translated_text(self):
        moment = ClipMoment(15, 45, "moment")
        words = [Word(15.2, 16.0, "Hello"), Word(16.1, 16.8, "there.")]

        result = make_srt(words, moment, ["こんにちは。"])

        self.assertIn("00:00:00,200 --> 00:00:01,800", result)
        self.assertIn("こんにちは。", result)
        self.assertNotIn("Hello", result)

    def test_translation_keeps_each_caption_aligned_to_its_clip(self):
        words = [
            Word(10.0, 10.5, "Hello"),
            Word(10.6, 11.0, "there."),
            Word(12.0, 12.5, "Goodbye."),
        ]
        moment = ClipMoment(10.0, 50.0, "moment")
        response = json.dumps({
            "translations": [
                {"id": 0, "text": "こんにちは。"},
                {"id": 1, "text": "さようなら。"},
            ]
        })

        with patch("video_clipper._ollama_chat", return_value=response):
            translations = translate_captions(words, [moment], "Japanese")

        self.assertEqual(translations, {0: ["こんにちは。", "さようなら。"]})

    def test_ffmpeg_render_burns_translated_captions_into_vertical_video(self):
        words = [Word(10.2, 11.0, "Hello there.")]
        moment = ClipMoment(10.0, 50.0, "moment")
        command = []

        def fake_ffmpeg(args, **_kwargs):
            command.extend(args)
            Path(args[-1]).write_bytes(b"rendered")
            return SimpleNamespace(returncode=0, stderr="")

        with TemporaryDirectory() as directory:
            with patch("video_clipper.subprocess.run", side_effect=fake_ffmpeg):
                output = render_clip(
                    "ffmpeg",
                    Path("source.mp4"),
                    Path(directory),
                    words,
                    moment,
                    "clip",
                    ["こんにちは。"],
                    "Japanese",
                )

            self.assertTrue(output.video_path.is_file())
            video_filter = command[command.index("-vf") + 1]
            self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase", video_filter)
            self.assertIn("subtitles=filename=", video_filter)
            self.assertIn("FontName=Hiragino Sans", video_filter)
            self.assertIn("こんにちは。", output.subtitle_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
