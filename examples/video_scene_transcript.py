#!/usr/bin/env python3
"""
Video Scene Transcript - Turn an MP4 into an illustrated, scene-by-scene Markdown document.

This script processes a single MP4 video through three stages:

1. **Transcription** with `WhisperX` - produces segment- and word-level
   timestamps for the spoken audio.
2. **Scene detection** with `PySceneDetect` - splits the video into visual
   scenes based on content changes.
3. **Composition** - grabs one representative screenshot per scene, matches the
   transcript segments that fall inside each scene, and writes everything out as
   a Markdown file with the images embedded.

The result is a `<video>.md` next to the images, ready to read or publish.

Requirements (not part of content-core's core dependencies):

    pip install whisperx "scenedetect[opencv]"

`ffmpeg` must also be available on the system PATH.

Usage:

    python examples/video_scene_transcript.py input.mp4
    python examples/video_scene_transcript.py input.mp4 --output-dir out --model small --language en
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("video_scene_transcript")


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging using the Python standard library.

    Uses only stdlib ``logging`` so the script stays dependency-free apart from
    WhisperX and PySceneDetect, which makes it easy to run in throwaway
    environments such as Google Colab.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def format_timestamp(seconds: float) -> str:
    """Format a number of seconds as HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _allow_full_torch_load() -> None:
    """Make ``torch.load`` default to ``weights_only=False`` for this process.

    PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``,
    which rejects the pickled objects (e.g. ``omegaconf.ListConfig``) inside the
    pyannote VAD and alignment checkpoints that WhisperX loads, raising an
    ``UnpicklingError``. These checkpoints come from the official, trusted
    WhisperX/pyannote model repos, so we restore the pre-2.6 behaviour by
    defaulting ``weights_only`` back to ``False``. The patch is idempotent and a
    no-op on older PyTorch versions.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with whisperx
        return

    if getattr(torch.load, "_full_load_patched", False):
        return

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    _patched_load._full_load_patched = True
    torch.load = _patched_load


def transcribe_video(
    input_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    batch_size: int,
) -> List[Dict]:
    """
    Transcribe the audio track of a video with WhisperX and return aligned segments.

    Each returned segment is a dict with at least ``start``, ``end`` and ``text``
    keys. Alignment is attempted to tighten the segment timestamps; if the
    alignment model cannot be loaded for the detected language, the raw
    transcription segments are returned instead.

    Args:
        input_path: Path to the input video file.
        model_size: WhisperX/Whisper model size (e.g. ``tiny``, ``base``, ``small``, ``large-v3``).
        device: Torch device, ``cuda`` or ``cpu``.
        compute_type: Faster-whisper compute type (e.g. ``float16``, ``int8``).
        language: Optional ISO language code. When ``None`` the language is auto-detected.
        batch_size: Batch size for transcription.

    Returns:
        A list of transcript segment dicts, ordered by start time.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise SystemExit(
            "WhisperX is not installed. Install it with:\n"
            '    pip install whisperx "scenedetect[opencv]"'
        ) from exc

    _allow_full_torch_load()

    logger.info(f"Loading WhisperX model '{model_size}' on {device} ({compute_type})")
    model = whisperx.load_model(
        model_size, device, compute_type=compute_type, language=language
    )

    logger.info("Loading audio and transcribing...")
    audio = whisperx.load_audio(str(input_path))
    result = model.transcribe(audio, batch_size=batch_size)
    detected_language = result.get("language", language)
    segments = result.get("segments", [])
    logger.info(
        f"Transcribed {len(segments)} raw segments (language: {detected_language})"
    )

    # Align for more accurate word/segment timestamps.
    try:
        logger.info("Aligning transcript for precise timestamps...")
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        segments = aligned.get("segments", segments)
        logger.info("Alignment complete")
    except Exception as exc:  # pragma: no cover - alignment is best-effort
        logger.warning(
            f"Alignment unavailable for language '{detected_language}' ({exc}); "
            "using unaligned segments"
        )

    # Keep only segments that carry actual text and valid timestamps.
    clean_segments = [
        {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", seg.get("start", 0.0))),
            "text": str(seg.get("text", "")).strip(),
        }
        for seg in segments
        if str(seg.get("text", "")).strip()
    ]
    clean_segments.sort(key=lambda s: s["start"])
    return clean_segments


def detect_scenes_and_screenshots(
    input_path: Path,
    output_dir: Path,
    threshold: float,
    min_scene_len: int,
) -> List[Dict]:
    """
    Detect scenes with PySceneDetect and save one screenshot per scene.

    Args:
        input_path: Path to the input video file.
        output_dir: Directory where scene screenshots are written.
        threshold: Content detector threshold - lower detects more scenes.
        min_scene_len: Minimum scene length in frames.

    Returns:
        A list of scene dicts with ``index`` (1-based), ``start`` and ``end``
        (seconds), and ``image`` (screenshot path relative to ``output_dir``,
        or ``None`` if no image could be saved).
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
        from scenedetect.scene_manager import save_images
    except ImportError as exc:
        raise SystemExit(
            "PySceneDetect is not installed. Install it with:\n"
            '    pip install whisperx "scenedetect[opencv]"'
        ) from exc

    logger.info(f"Detecting scenes (threshold={threshold}, min_scene_len={min_scene_len})")
    video = open_video(str(input_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    # If no cuts were found the whole video is a single scene.
    if not scene_list:
        logger.info("No scene cuts detected; treating the whole video as one scene")
        duration = video.duration.get_seconds() if video.duration else 0.0
        return [{"index": 1, "start": 0.0, "end": duration, "image": None}]

    logger.info(f"Detected {len(scene_list)} scenes; saving screenshots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # One screenshot per scene, taken from the middle of the scene.
    image_map = save_images(
        scene_list,
        video,
        num_images=1,
        image_name_template="scene-$SCENE_NUMBER",
        output_dir=str(output_dir),
    )

    scenes: List[Dict] = []
    for i, (start_time, end_time) in enumerate(scene_list):
        images = image_map.get(i, [])
        scenes.append(
            {
                "index": i + 1,
                "start": start_time.get_seconds(),
                "end": end_time.get_seconds(),
                "image": images[0] if images else None,
            }
        )
    return scenes


def assign_segments_to_scenes(
    scenes: List[Dict], segments: List[Dict]
) -> Dict[int, List[Dict]]:
    """
    Assign each transcript segment to the scene that contains its midpoint.

    Using the segment midpoint guarantees every segment lands in exactly one
    scene, so no text is duplicated or dropped. Segments whose midpoint falls
    past the last scene boundary are attached to the final scene.

    Args:
        scenes: Scene dicts as produced by :func:`detect_scenes_and_screenshots`.
        segments: Transcript segment dicts with ``start``/``end``/``text``.

    Returns:
        A mapping of scene index (1-based) to its ordered list of segments.
    """
    buckets: Dict[int, List[Dict]] = {scene["index"]: [] for scene in scenes}

    for segment in segments:
        midpoint = (segment["start"] + segment["end"]) / 2.0
        target = scenes[-1]  # default: trailing audio belongs to the last scene
        for scene in scenes:
            if scene["start"] <= midpoint < scene["end"]:
                target = scene
                break
        buckets[target["index"]].append(segment)

    return buckets


def build_markdown(
    input_path: Path,
    output_dir: Path,
    scenes: List[Dict],
    segments_by_scene: Dict[int, List[Dict]],
) -> str:
    """
    Compose the final Markdown document with embedded scene screenshots.

    Image links are written relative to ``output_dir`` so the Markdown file and
    its images stay portable as long as they travel together.
    """
    lines: List[str] = []
    lines.append(f"# {input_path.stem}")
    lines.append("")
    lines.append(f"*Source: `{input_path.name}` — {len(scenes)} scenes*")
    lines.append("")

    for scene in scenes:
        start = format_timestamp(scene["start"])
        end = format_timestamp(scene["end"])
        lines.append(f"## Scene {scene['index']} — {start} → {end}")
        lines.append("")

        if scene["image"]:
            image_path = Path(scene["image"])
            # `save_images` returns paths relative to output_dir; keep them relative.
            rel = image_path.name if image_path.is_absolute() else scene["image"]
            lines.append(f"![Scene {scene['index']}]({rel})")
            lines.append("")

        scene_segments = segments_by_scene.get(scene["index"], [])
        if scene_segments:
            transcript = " ".join(seg["text"] for seg in scene_segments).strip()
            lines.append(transcript)
        else:
            lines.append("*(no speech in this scene)*")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def process_video(
    input_path: Path,
    output_dir: Path,
    model_size: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    batch_size: int,
    threshold: float,
    min_scene_len: int,
) -> Path:
    """Run the full transcribe → detect → compose pipeline for one video."""
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    segments = transcribe_video(
        input_path=input_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )

    scenes = detect_scenes_and_screenshots(
        input_path=input_path,
        output_dir=output_dir,
        threshold=threshold,
        min_scene_len=min_scene_len,
    )

    segments_by_scene = assign_segments_to_scenes(scenes, segments)

    markdown = build_markdown(input_path, output_dir, scenes, segments_by_scene)

    output_md = output_dir / f"{input_path.stem}.md"
    output_md.write_text(markdown, encoding="utf-8")
    logger.info(f"Wrote {output_md} ({len(scenes)} scenes)")
    return output_md


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an MP4 with WhisperX, detect scenes with PySceneDetect, "
            "and produce a Markdown file with a screenshot and transcript per scene."
        )
    )
    parser.add_argument("video", type=Path, help="Path to the input MP4 video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for screenshots and the Markdown file "
        "(default: <video-name>_scenes next to the video)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="WhisperX model size: tiny, base, small, medium, large-v3 (default: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code (e.g. en, nl). Default: auto-detect",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type, e.g. int8 (cpu), float16 (cuda) (default: int8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Transcription batch size (default: 8)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="PySceneDetect content threshold; lower = more scenes (default: 27.0)",
    )
    parser.add_argument(
        "--min-scene-len",
        type=int,
        default=15,
        help="Minimum scene length in frames (default: 15)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    input_path: Path = args.video
    output_dir: Path = args.output_dir or input_path.with_name(
        f"{input_path.stem}_scenes"
    )

    logger.info(f"Processing {input_path} → {output_dir}")
    process_video(
        input_path=input_path,
        output_dir=output_dir,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
    )


if __name__ == "__main__":
    main()
