"""Orquestracao quadro a quadro e por janela temporal."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .models import ActionClassifier, Detector, Segmenter
from .processing import crop_frames, prediction_for_frame, project_mask, select_primary_person
from .types import ActionPrediction, BoundingBox, Detection


class VideoPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        detector: Detector,
        segmenter: Segmenter,
        classifier: ActionClassifier,
    ) -> None:
        self.config = config
        self.detector = detector
        self.segmenter = segmenter
        self.classifier = classifier

    def _annotate(
        self,
        frame: np.ndarray,
        box: BoundingBox | None,
        detection_confidence: float,
        mask: np.ndarray | None,
        action: tuple[str, float] | None,
        show_action: bool = True,
    ) -> np.ndarray:
        import cv2

        output = frame.copy()
        if mask is not None and mask.any():
            color = np.asarray(self.config.output.mask_color_bgr, dtype=np.uint8)
            overlay = np.broadcast_to(color, output.shape)
            alpha = self.config.output.mask_alpha
            output[mask] = ((1 - alpha) * output[mask] + alpha * overlay[mask]).astype(np.uint8)
        if box is not None:
            color = tuple(self.config.output.box_color_bgr)
            cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            cv2.putText(
                output,
                f"pessoa {detection_confidence:.0%}",
                (box.x1, max(24, box.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        if not show_action:
            return output
        label, confidence = action or ("sem predicao", 0.0)
        action_text = f"acao: {label} | confianca: {confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.70
        base_width = cv2.getTextSize(action_text, font, base_scale, 2)[0][0]
        available_width = max(1, output.shape[1] - 24)
        font_scale = max(0.40, min(base_scale, base_scale * available_width / max(1, base_width)))
        thickness = 1 if font_scale < 0.52 else 2
        cv2.rectangle(output, (0, 0), (output.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(
            output,
            action_text,
            (12, 28),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return output

    def run(self, input_path: str | Path, output_dir: str | Path) -> dict[str, object]:
        import cv2

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Nao foi possivel abrir o video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < total_frames < self.config.slowfast.window_size:
            capture.release()
            raise ValueError(
                f"O video possui {total_frames} quadros; SlowFast requer ao menos "
                f"{self.config.slowfast.window_size}."
            )
        video_path = output_dir / f"{input_path.stem}_anotado.mp4"
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".mp4", delete=False) as temporary:
            spatial_path = Path(temporary.name)
        writer = cv2.VideoWriter(
            str(spatial_path),
            cv2.VideoWriter_fourcc(*self.config.output.codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Nao foi possivel criar o video: {video_path}")

        clip_frames: deque[np.ndarray] = deque(maxlen=self.config.slowfast.window_size)
        clip_boxes: deque[BoundingBox | None] = deque(maxlen=self.config.slowfast.window_size)
        predictions: list[ActionPrediction] = []
        frame_rows: list[dict[str, object]] = []
        frame_index = 0
        previous_box: BoundingBox | None = None

        def classify_window(end_frame: int) -> None:
            frames_for_action = list(clip_frames)
            if self.config.slowfast.person_crop:
                frames_for_action = crop_frames(frames_for_action, list(clip_boxes))
            ranked = self.classifier.classify(frames_for_action)
            if not ranked:
                raise RuntimeError("SlowFast nao retornou nenhuma classe.")
            predictions.append(ActionPrediction(
                label=ranked[0][0], confidence=ranked[0][1],
                start_frame=end_frame + 1 - self.config.slowfast.window_size,
                end_frame=end_frame, ranked=tuple(ranked),
            ))

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = self.detector.detect(frame)
                segmented = []
                for detection in detections:
                    candidate_box = detection.box.clip(width, height)
                    if candidate_box.area == 0:
                        continue
                    expanded = candidate_box.expand(self.config.unet.box_margin, width, height)
                    crop = frame[expanded.y1 : expanded.y2, expanded.x1 : expanded.x2]
                    candidate_mask = project_mask(self.segmenter.segment(crop), expanded, frame.shape)
                    segmented.append(Detection(candidate_box, detection.confidence, candidate_mask))
                # A U-Net e executada nos recortes YOLO. Preferir uma mascara
                # valida evita persistir em uma pessoa escura ao fundo quando
                # ha outro candidato segmentavel; nao garante identidade.
                with_mask = [item for item in segmented if item.mask is not None and item.mask.any()]
                primary = select_primary_person(with_mask or segmented, previous_box)
                box = None
                mask = None
                det_conf = 0.0
                if primary is not None:
                    box = primary.box.clip(width, height)
                    previous_box = box
                    det_conf = primary.confidence
                    mask = primary.mask
                clip_frames.append(frame.copy())
                clip_boxes.append(box)
                ready = len(clip_frames) == self.config.slowfast.window_size
                due = ready and (frame_index + 1 - self.config.slowfast.window_size) % self.config.slowfast.stride == 0
                if due:
                    classify_window(frame_index)
                writer.write(self._annotate(frame, box, det_conf, mask, None, show_action=False))
                frame_rows.append(
                    {
                        "frame": frame_index,
                        "tempo_s": round(frame_index / fps, 3),
                        "pessoa_detectada": box is not None,
                        "confianca_yolo": round(det_conf, 5),
                        "area_mascara_px": int(mask.sum()) if mask is not None else 0,
                        "numero_pessoas": len(detections),
                        "candidatos_mascara_naovazia": len(with_mask),
                        "bbox_x1": box.x1 if box else "",
                        "bbox_y1": box.y1 if box else "",
                        "bbox_x2": box.x2 if box else "",
                        "bbox_y2": box.y2 if box else "",
                    }
                )
                frame_index += 1
            if frame_index < self.config.slowfast.window_size:
                raise ValueError("O video decodificado nao possui quadros suficientes para SlowFast.")
            # Fecha a cauda mesmo quando o numero de quadros nao e multiplo do stride.
            if not predictions or predictions[-1].end_frame != frame_index - 1:
                classify_window(frame_index - 1)
        except Exception:
            spatial_path.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            writer.release()

        # Segunda passagem leve: a predicao e mostrada sobre os quadros da
        # propria janela, inclusive o inicio. Nao simula inferencia em tempo real.
        spatial_capture = cv2.VideoCapture(str(spatial_path))
        final_writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*self.config.output.codec), fps, (width, height),
        )
        try:
            if not spatial_capture.isOpened() or not final_writer.isOpened():
                raise RuntimeError("Falha ao abrir a segunda passagem do video.")
            for index, row in enumerate(frame_rows):
                ok, annotated = spatial_capture.read()
                if not ok:
                    raise RuntimeError(f"Video intermediario incompleto no quadro {index}.")
                prediction = prediction_for_frame(predictions, index)
                row.update({
                    "acao": prediction.label,
                    "confianca_slowfast": round(prediction.confidence, 5),
                    "frame_inicial_janela": prediction.start_frame,
                    "frame_final_janela": prediction.end_frame,
                })
                final_writer.write(self._annotate(
                    annotated, None, 0.0, None, (prediction.label, prediction.confidence),
                ))
        finally:
            spatial_capture.release()
            final_writer.release()
            spatial_path.unlink(missing_ok=True)

        csv_path = output_dir / f"{input_path.stem}_frames.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(frame_rows[0].keys()) if frame_rows else [])
            if frame_rows:
                writer_csv.writeheader()
                writer_csv.writerows(frame_rows)
        json_path = output_dir / f"{input_path.stem}_janelas.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "acao": item.label,
                        "confianca": item.confidence,
                        "frame_inicial": item.start_frame,
                        "frame_final": item.end_frame,
                        "inicio_s": round(item.start_frame / fps, 3),
                        "fim_s": round((item.end_frame + 1) / fps, 3),
                        "top_k": [{"acao": label, "confianca": score} for label, score in item.ranked],
                    }
                    for item in predictions
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "input": str(input_path),
            "video": str(video_path),
            "csv": str(csv_path),
            "json": str(json_path),
            "frames": frame_index,
            "fps": fps,
            "frames_declarados": total_frames,
        }
