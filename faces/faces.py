from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace


ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceMatch:
    face_ratio: float
    similarity: float
    detection_score: float


@dataclass
class FrameResult:
    frame_path: Path
    frame_score: float
    detected_target: bool
    face_ratio: float
    similarity: float
    sharpness: float
    detection_score: float
    face_count: int


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported device: {device}")

    if normalized == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")
    if normalized == "gpu":
        if not gpus:
            raise ValueError("GPU device requested, but TensorFlow did not detect an available GPU.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return "/GPU:0"

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return "/GPU:0"

    return "/CPU:0"


def _load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _landmarks_to_array(landmarks: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks["nose"],
            landmarks["mouth_left"],
            landmarks["mouth_right"],
        ],
        dtype=np.float32,
    )


def _align_face(image: np.ndarray, landmarks: dict[str, Any], size: tuple[int, int] = (112, 112)) -> np.ndarray:
    source_points = _landmarks_to_array(landmarks)
    target_points = ARCFACE_TEMPLATE.copy()
    target_points[:, 0] *= size[0] / 112.0
    target_points[:, 1] *= size[1] / 112.0

    transform, _ = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.LMEDS,
    )
    if transform is None:
        x1, y1, x2, y2 = _landmark_box(landmarks, image.shape[1], image.shape[0])
        return cv2.resize(image[y1:y2, x1:x2], size)

    return cv2.warpAffine(image, transform, size, borderMode=cv2.BORDER_REPLICATE)


def _safe_box(face: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    area = face["facial_area"]
    x1 = max(int(area[0]), 0)
    y1 = max(int(area[1]), 0)
    x2 = min(int(area[2]), width)
    y2 = min(int(area[3]), height)
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)
    return x1, y1, x2, y2


def _landmark_box(landmarks: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    points = _landmarks_to_array(landmarks)
    x1 = max(int(np.min(points[:, 0])) - 20, 0)
    y1 = max(int(np.min(points[:, 1])) - 20, 0)
    x2 = min(int(np.max(points[:, 0])) + 20, width)
    y2 = min(int(np.max(points[:, 1])) + 20, height)
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)
    return x1, y1, x2, y2


def _compute_embedding(face_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    intensity = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.resize(grad_mag, (32, 32), interpolation=cv2.INTER_AREA)
    grad_mag /= float(np.max(grad_mag) + 1e-6)

    ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
    chroma = cv2.resize(ycrcb[:, :, 1:], (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    embedding = np.concatenate(
        [
            intensity.flatten(),
            grad_mag.flatten(),
            chroma.flatten(),
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(embedding))
    if norm == 0:
        return embedding
    return embedding / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def _frame_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_ratio(face: dict[str, Any], image_shape: tuple[int, ...]) -> float:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = _safe_box(face, width, height)
    area = max((x2 - x1) * (y2 - y1), 0)
    return float(area / max(width * height, 1))


def _detect_faces(
    image: np.ndarray,
    model: Any,
    detection_threshold: float,
    tf_device: str,
) -> list[dict[str, Any]]:
    with tf.device(tf_device):
        detections = RetinaFace.detect_faces(
            img_path=image,
            threshold=detection_threshold,
            model=model,
        )
    if not isinstance(detections, dict):
        return []
    return sorted(detections.values(), key=lambda item: float(item["score"]), reverse=True)


def _build_model(tf_device: str) -> Any:
    with tf.device(tf_device):
        return RetinaFace.build_model()


def load_target_embedding(
    target_face_path: str | Path,
    model: Any | None = None,
    detection_threshold: float = 0.9,
    device: str = "auto",
    tf_device: str | None = None,
) -> tuple[np.ndarray, Any]:
    active_tf_device = tf_device or _resolve_device(device)
    active_model = model or _build_model(active_tf_device)
    image = _load_image(target_face_path)
    detections = _detect_faces(image, active_model, detection_threshold, active_tf_device)
    if not detections:
        raise ValueError(f"No face detected in target image: {target_face_path}")

    target_face = max(detections, key=lambda face: _face_ratio(face, image.shape))
    aligned_face = _align_face(image, target_face["landmarks"])
    return _compute_embedding(aligned_face), active_model


def analyze_frame(
    frame_path: str | Path,
    target_embedding: np.ndarray,
    model: Any,
    similarity_threshold: float = 0.75,
    detection_threshold: float = 0.9,
    device: str = "auto",
    tf_device: str | None = None,
) -> FrameResult:
    frame = _load_image(frame_path)
    active_tf_device = tf_device or _resolve_device(device)
    detections = _detect_faces(frame, model, detection_threshold, active_tf_device)
    sharpness = _frame_sharpness(frame)

    best_match = FaceMatch(face_ratio=0.0, similarity=0.0, detection_score=0.0)
    for face in detections:
        aligned_face = _align_face(frame, face["landmarks"])
        embedding = _compute_embedding(aligned_face)
        similarity = _cosine_similarity(target_embedding, embedding)
        match = FaceMatch(
            face_ratio=_face_ratio(face, frame.shape),
            similarity=similarity,
            detection_score=float(face["score"]),
        )
        if match.similarity > best_match.similarity:
            best_match = match

    detected_target = best_match.similarity >= similarity_threshold
    frame_score = (
        10 * int(detected_target)
        + 80 * best_match.face_ratio
        + 5 * best_match.similarity
        + min(sharpness / 200.0, 5.0)
    )

    return FrameResult(
        frame_path=Path(frame_path),
        frame_score=frame_score,
        detected_target=detected_target,
        face_ratio=best_match.face_ratio,
        similarity=best_match.similarity,
        sharpness=sharpness,
        detection_score=best_match.detection_score,
        face_count=len(detections),
    )


def analyze_frames(
    target_face_path: str | Path,
    frames_dir: str | Path,
    result_file: str | Path,
    similarity_threshold: float = 0.75,
    detection_threshold: float = 0.9,
    device: str = "auto",
) -> list[FrameResult]:
    frames_root = Path(frames_dir)
    frame_paths = sorted(path for path in frames_root.glob("*.jpg") if path.is_file())
    if not frame_paths:
        raise ValueError(f"No jpg frames found in directory: {frames_root}")

    tf_device = _resolve_device(device)
    target_embedding, model = load_target_embedding(
        target_face_path=target_face_path,
        detection_threshold=detection_threshold,
        device=device,
        tf_device=tf_device,
    )

    results = [
        analyze_frame(
            frame_path=frame_path,
            target_embedding=target_embedding,
            model=model,
            similarity_threshold=similarity_threshold,
            detection_threshold=detection_threshold,
            device=device,
            tf_device=tf_device,
        )
        for frame_path in frame_paths
    ]
    results.sort(key=lambda item: item.frame_score, reverse=True)

    output_path = Path(result_file)
    lines = [
        "# frame_path\tscore\tdetected_target\tface_ratio\tsimilarity\tsharpness\tface_count\tdetection_score"
    ]
    for result in results:
        lines.append(
            "\t".join(
                [
                    str(result.frame_path),
                    f"{result.frame_score:.6f}",
                    str(int(result.detected_target)),
                    f"{result.face_ratio:.6f}",
                    f"{result.similarity:.6f}",
                    f"{result.sharpness:.6f}",
                    str(result.face_count),
                    f"{result.detection_score:.6f}",
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results
