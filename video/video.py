from __future__ import annotations

from pathlib import Path

import cv2


def extract_frames_one_per_second(video_path: str | Path, output_dir: str | Path) -> int:
    """Read a video with OpenCV and save one frame per second."""
    source = Path(video_path)
    target_dir = Path(output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Unable to read FPS from video: {source}")

    saved_count = 0
    next_capture_time = 0.0
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            current_time = frame_index / fps
            if current_time + 1e-9 >= next_capture_time:
                output_path = target_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
                next_capture_time += 1.0

            frame_index += 1
    finally:
        capture.release()

    return saved_count
