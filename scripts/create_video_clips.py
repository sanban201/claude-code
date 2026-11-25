#!/usr/bin/env python3
"""Create up to three subtitled 9:16 clips from a video in ``videos/``.

Install ``openai-whisper``, ``moviepy`` and ``pillow``, and make sure ``ffmpeg``
is on PATH. Run ``python scripts/create_video_clips.py`` to use the only video
in ``videos/``, or pass a filename/path with ``--input``.

Clip ranking is a transcript-based heuristic (speech density, vocabulary
variety, emphasis, and engaging words), not a learned measure of audience
engagement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from moviepy import CompositeVideoClip, ImageClip, VideoFileClip
except ImportError:
    from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip

import numpy as np
import whisper
from PIL import Image, ImageDraw, ImageFont


CLIP_SECONDS = 45.0
CLIP_COUNT = 3
OUTPUT_SIZE = (1080, 1920)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ENGAGING_WORDS = {
    "amazing", "because", "best", "crazy", "discover", "finally", "important",
    "imagine", "incredible", "mistake", "never", "secret", "surprising",
    "wait", "warning", "why",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Video filename in videos/ or a video path")
    parser.add_argument("--videos-dir", type=Path, default=Path("videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--device", help="Whisper device, such as cpu, cuda, or mps")
    return parser.parse_args()


def resolve_input(value: str | None, videos_dir: Path) -> Path:
    if value:
        supplied = Path(value).expanduser()
        candidates = [supplied] if supplied.is_absolute() else [supplied, videos_dir / supplied]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Video not found: {value}")

    videos = sorted(
        path for path in videos_dir.glob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No supported video files found in {videos_dir}")
    if len(videos) > 1:
        names = ", ".join(path.name for path in videos)
        raise ValueError(f"Found multiple videos in {videos_dir}; choose one with --input ({names})")
    return videos[0].resolve()


def extract_words(transcription: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcription.get("segments", []):
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                text = str(word.get("word", "")).strip()
                if text and word.get("start") is not None and word.get("end") is not None:
                    words.append({
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": text,
                    })
            continue

        text_words = str(segment.get("text", "")).split()
        start, end = float(segment["start"]), float(segment["end"])
        if text_words and end > start:
            step = (end - start) / len(text_words)
            words.extend(
                {"start": start + i * step, "end": start + (i + 1) * step, "text": word}
                for i, word in enumerate(text_words)
            )
    return sorted(words, key=lambda word: word["start"])


def score_window(words: list[dict[str, Any]], start: float, end: float) -> float:
    included = [
        word for word in words
        if word["end"] > start and word["start"] < end
    ]
    if not included:
        return 0.0

    # Estimate spoken coverage from the clipped word intervals.
    intervals = sorted(
        (max(start, word["start"]), min(end, word["end"]))
        for word in included
    )
    spoken = 0.0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            spoken += current_end - current_start
            current_start, current_end = interval_start, interval_end
    spoken += current_end - current_start

    tokens = [re.sub(r"[^a-z0-9']", "", word["text"].lower()) for word in included]
    tokens = [token for token in tokens if token]
    if not tokens:
        return 0.0

    density = len(tokens) / CLIP_SECONDS
    coverage = min(spoken / CLIP_SECONDS, 1.0)
    variety = len(set(tokens)) / len(tokens)
    emphasis = sum(
        word["text"].count("!") + word["text"].count("?")
        for word in included
    ) / len(included)
    hooks = sum(token in ENGAGING_WORDS for token in tokens) / len(tokens)
    return coverage * 4.0 + min(density, 3.0) + variety + emphasis * 2.0 + hooks * 5.0


def choose_segments(words: list[dict[str, Any]], duration: float) -> list[tuple[float, float]]:
    if duration < CLIP_SECONDS:
        raise ValueError(f"Video must be at least {CLIP_SECONDS:g} seconds long")

    step = 5.0
    last_start = duration - CLIP_SECONDS
    starts = [float(index) * step for index in range(int(last_start // step) + 1)]
    if not starts or starts[-1] < last_start:
        starts.append(last_start)
    ranked = sorted(
        ((score_window(words, start, start + CLIP_SECONDS), start) for start in starts),
        key=lambda item: (-item[0], item[1]),
    )

    chosen: list[tuple[float, float]] = []
    for _, start in ranked:
        end = start + CLIP_SECONDS
        if all(end <= chosen_start or start >= chosen_end for chosen_start, chosen_end in chosen):
            chosen.append((start, end))
            if len(chosen) == CLIP_COUNT:
                break
    return sorted(chosen)


def subtitle_groups(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    clip_words = [
        {
            "start": max(start, word["start"]) - start,
            "end": min(end, word["end"]) - start,
            "text": word["text"],
        }
        for word in words if word["end"] > start and word["start"] < end
    ]
    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for word in clip_words:
        text_length = len(word["text"])
        if current and (len(current) >= 5 or current_chars + text_length > 32):
            groups.append(_make_group(current))
            current, current_chars = [], 0
        current.append(word)
        current_chars += text_length
    if current:
        groups.append(_make_group(current))
    return groups


def _make_group(words: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": words[0]["start"],
        "end": max(words[-1]["end"], words[0]["start"] + 0.15),
        "text": " ".join(word["text"] for word in words).strip(),
    }


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def subtitle_image(text: str) -> np.ndarray:
    width, _ = OUTPUT_SIZE
    canvas_height = 250
    image = Image.new("RGBA", (width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = load_font(58)
    max_width = width - 120

    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    lines = lines[-2:]

    line_heights = [
        draw.textbbox((0, 0), value, font=font, stroke_width=2)[3]
        - draw.textbbox((0, 0), value, font=font, stroke_width=2)[1]
        for value in lines
    ]
    padding_x, padding_y, gap = 28, 18, 12
    box_width = max(
        draw.textbbox((0, 0), value, font=font, stroke_width=2)[2]
        for value in lines
    ) + padding_x * 2
    box_height = sum(line_heights) + gap * (len(lines) - 1) + padding_y * 2
    box_x = (width - box_width) // 2
    box_y = 0
    draw.rounded_rectangle(
        (box_x, box_y, box_x + box_width, box_y + box_height),
        radius=18,
        fill=(0, 0, 0, 170),
    )

    y = box_y + padding_y
    for value, line_height in zip(lines, line_heights):
        text_width = draw.textbbox((0, 0), value, font=font, stroke_width=2)[2]
        draw.text(
            ((width - text_width) / 2, y),
            value,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        y += line_height + gap
    return np.asarray(image)


def _subclip(clip: Any, start: float, end: float) -> Any:
    method = getattr(clip, "subclipped", None) or clip.subclip
    return method(start, end)


def _crop(clip: Any, x1: float, y1: float, x2: float, y2: float) -> Any:
    if hasattr(clip, "cropped"):
        return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


def _resize(clip: Any, size: tuple[int, int]) -> Any:
    if hasattr(clip, "resized"):
        return clip.resized(new_size=size)
    return clip.resize(newsize=size)


def _with_timing(clip: Any, start: float, duration: float) -> Any:
    if hasattr(clip, "with_start"):
        return clip.with_start(start).with_duration(duration)
    return clip.set_start(start).set_duration(duration)


def _with_position(clip: Any, position: tuple[str, str]) -> Any:
    if hasattr(clip, "with_position"):
        return clip.with_position(position)
    return clip.set_position(position)


def render_segment(
    source: Any,
    words: list[dict[str, Any]],
    start: float,
    end: float,
    destination: Path,
) -> None:
    width, height = source.size
    target_ratio = OUTPUT_SIZE[0] / OUTPUT_SIZE[1]
    if width / height > target_ratio:
        crop_width = height * target_ratio
        x1, x2 = (width - crop_width) / 2, (width + crop_width) / 2
        y1, y2 = 0, height
    else:
        crop_height = width / target_ratio
        x1, x2 = 0, width
        y1, y2 = (height - crop_height) / 2, (height + crop_height) / 2

    video = _resize(
        _crop(_subclip(source, start, end), x1, y1, x2, y2),
        OUTPUT_SIZE,
    )
    duration = end - start
    overlays = [
        _with_timing(
            _with_position(ImageClip(subtitle_image(group["text"])), ("center", "bottom")),
            group["start"],
            min(group["end"] - group["start"], duration - group["start"]),
        )
        for group in subtitle_groups(words, start, end)
        if group["end"] > group["start"] and group["start"] < duration
    ]
    final = CompositeVideoClip([video, *overlays], size=OUTPUT_SIZE)
    try:
        fps = getattr(source, "fps", None) or 30
        final.write_videofile(
            str(destination),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
    finally:
        final.close()


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input(args.input, args.videos_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        model = whisper.load_model(args.model, device=args.device)
        transcription = model.transcribe(
            str(input_path),
            word_timestamps=True,
            verbose=False,
        )
        words = extract_words(transcription)
        with (args.output_dir / f"{input_path.stem}_transcript.json").open(
            "w", encoding="utf-8"
        ) as transcript_file:
            json.dump(transcription, transcript_file, ensure_ascii=False, indent=2)

        with VideoFileClip(str(input_path)) as source:
            segments = choose_segments(words, float(source.duration))
            if len(segments) < CLIP_COUNT:
                print(
                    f"Note: only {len(segments)} non-overlapping 45-second segment(s) "
                    "fit this video.",
                    file=sys.stderr,
                )
            for index, (start, end) in enumerate(segments, start=1):
                destination = args.output_dir / f"{input_path.stem}_clip_{index:02d}.mp4"
                render_segment(source, words, start, end, destination)
                print(f"Saved {destination} ({start:.1f}-{end:.1f}s)")
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
