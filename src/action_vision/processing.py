"""Operacoes puras de geometria, amostragem e visualizacao."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import ActionPrediction, BoundingBox, Detection


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def select_primary_person(
    detections: Sequence[Detection], previous_box: BoundingBox | None = None,
) -> Detection | None:
    """Seleciona a pessoa combinando confianca e area para reduzir trocas."""

    if not detections:
        return None
    if previous_box is not None:
        overlapping = [item for item in detections if box_iou(item.box, previous_box) >= 0.20]
        if overlapping:
            return max(overlapping, key=lambda item: box_iou(item.box, previous_box) * item.confidence)
    return max(detections, key=lambda detection: detection.confidence * np.sqrt(detection.box.area))


def prediction_for_frame(predictions: Sequence[ActionPrediction], index: int) -> ActionPrediction:
    """No video offline, atribui a janela que cobre o quadro e tem centro mais proximo."""

    covering = [item for item in predictions if item.start_frame <= index <= item.end_frame]
    if not covering:
        raise ValueError(f"Nenhuma janela SlowFast cobre o quadro {index}.")
    return min(covering, key=lambda item: abs((item.start_frame + item.end_frame) / 2 - index))


def project_mask(mask: np.ndarray, box: BoundingBox, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Redimensiona uma mascara do recorte e a projeta no quadro original."""

    import cv2

    frame_height, frame_width = frame_shape[:2]
    clipped = box.clip(frame_width, frame_height)
    output = np.zeros((frame_height, frame_width), dtype=bool)
    if clipped.area == 0 or mask.size == 0:
        return output
    resized = cv2.resize(mask.astype(np.uint8), (clipped.width, clipped.height), interpolation=cv2.INTER_NEAREST)
    output[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2] = resized.astype(bool)
    return output


def crop_frames(frames: Sequence[np.ndarray], boxes: Sequence[BoundingBox | None]) -> list[np.ndarray]:
    """Gera recortes temporais com dimensoes fixas pela uniao das caixas."""

    valid = [box for box in boxes if box is not None and box.area > 0]
    if not valid:
        return list(frames)
    union = BoundingBox(
        min(box.x1 for box in valid),
        min(box.y1 for box in valid),
        max(box.x2 for box in valid),
        max(box.y2 for box in valid),
    )
    height, width = frames[0].shape[:2]
    union = union.expand(0.15, width, height)
    if union.area == 0:
        return list(frames)
    return [frame[union.y1 : union.y2, union.x1 : union.x2] for frame in frames]


def temporal_windows(total_frames: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Retorna janelas completas e cobre o fim do video sem duplicar intervalos."""

    if total_frames <= 0:
        return []
    if total_frames <= window_size:
        return [(0, total_frames)]
    starts = list(range(0, total_frames - window_size + 1, stride))
    last_start = total_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_size) for start in starts]

