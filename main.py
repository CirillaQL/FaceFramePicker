from __future__ import annotations

import argparse
from pathlib import Path

from faces.faces import analyze_frames
from video.video import extract_frames_one_per_second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one frame per second from a video.")
    parser.add_argument("video_path", nargs="?", help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory used to store extracted frames. Defaults to output/",
    )
    parser.add_argument(
        "--target-face",
        help="Path to the target face image used for scoring extracted frames.",
    )
    parser.add_argument(
        "--frames-dir",
        help="Directory containing extracted frame images. Defaults to output/ or --output-dir.",
    )
    parser.add_argument(
        "--result-file",
        default="1.txt",
        help="Path to the ranking result file. Defaults to 1.txt.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold used to mark the target face as detected.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.9,
        help="RetinaFace confidence threshold.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="TensorFlow device used by RetinaFace. Defaults to auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")
    frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir

    if args.video_path:
        video_path = Path(args.video_path)
        frame_count = extract_frames_one_per_second(video_path, output_dir)
        print(f"Saved {frame_count} frame(s) to {output_dir.resolve()}")

    if args.target_face:
        results = analyze_frames(
            target_face_path=args.target_face,
            frames_dir=frames_dir,
            result_file=args.result_file,
            similarity_threshold=args.similarity_threshold,
            detection_threshold=args.detection_threshold,
            device=args.device,
        )
        print(
            f"Ranked {len(results)} frame(s) from {frames_dir.resolve()} "
            f"and wrote results to {Path(args.result_file).resolve()}"
        )
    elif not args.video_path:
        raise SystemExit("Please provide a video path, --target-face, or both.")


if __name__ == "__main__":
    main()
