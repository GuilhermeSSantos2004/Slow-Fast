"""Executa YOLO, U-Net e SlowFast separadamente e compara quadro/recorte."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from action_vision.config import PipelineConfig
from action_vision.models import SlowFastClassifier, YoloPersonDetector, build_segmenter, resolve_device
from action_vision.pipeline import VideoPipeline
from action_vision.processing import (
    crop_frames,
    prediction_for_frame,
    project_mask,
    select_primary_person,
    temporal_windows,
)
from action_vision.types import ActionPrediction, BoundingBox


def run_individual(input_path, output_dir, config, detector, segmenter, classifier):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(input_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < config.slowfast.window_size:
        raise ValueError("O experimento requer pelo menos 32 quadros.")
    height, width = frames[0].shape[:2]
    stem = Path(input_path).stem
    pipeline = VideoPipeline(config, detector, segmenter, classifier)
    writers = {}
    for stage in ("yolo", "unet", "slowfast"):
        writer = cv2.VideoWriter(
            str(output_dir / f"{stem}_{stage}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Nao foi possivel criar video {stage}.")
        writers[stage] = writer
    rows = []
    boxes = []
    integrated_csv = output_dir.parent / f"{stem}_frames.csv"
    integrated_rows = None
    if integrated_csv.exists():
        with integrated_csv.open(encoding="utf-8") as stream:
            integrated_rows = list(csv.DictReader(stream))
        if len(integrated_rows) != len(frames):
            raise ValueError("CSV integrado e video possuem quantidades diferentes de quadros.")
    try:
        for index, frame in enumerate(frames):
            # U-Net isolada recebe o quadro inteiro, sem resultados de YOLO.
            detections = detector.detect(frame)
            primary = select_primary_person(detections)
            box = primary.box if primary else None
            if integrated_rows is not None:
                row = integrated_rows[index]
                box = BoundingBox(*[int(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]) if row["bbox_x1"] else None
            boxes.append(box)
            annotated = frame.copy()
            for detection in detections:
                annotated = pipeline._annotate(
                    annotated, detection.box, detection.confidence, None, None, show_action=False,
                )
            writers["yolo"].write(annotated)
            mask = project_mask(segmenter.segment(frame), BoundingBox(0, 0, width, height), frame.shape)
            writers["unet"].write(pipeline._annotate(frame, None, 0, mask, None, show_action=False))
            rows.append({
                "frame": index, "tempo_s": round(index / fps, 3), "pessoas_yolo": len(detections),
                "mascara_unet_px": int(mask.sum()),
            })
        predictions = []
        comparisons = []
        for start, stop in temporal_windows(len(frames), config.slowfast.window_size, config.slowfast.stride):
            ranked = classifier.classify(frames[start:stop])
            cropped = classifier.classify(crop_frames(frames[start:stop], boxes[start:stop]))
            predictions.append(ActionPrediction(ranked[0][0], ranked[0][1], start, stop - 1, tuple(ranked)))
            comparisons.append({
                "frame_inicial": start, "frame_final": stop - 1,
                "inicio_s": round(start / fps, 3), "fim_s": round(stop / fps, 3),
                "quadro_inteiro": [{"acao": label, "confianca": score} for label, score in ranked],
                "recorte_pessoa": [{"acao": label, "confianca": score} for label, score in cropped],
                "origem_caixas_recorte": "pipeline integrado" if integrated_rows is not None else "YOLO isolado",
            })
        for index, frame in enumerate(frames):
            prediction = prediction_for_frame(predictions, index)
            writers["slowfast"].write(pipeline._annotate(
                frame, None, 0, None, (prediction.label, prediction.confidence),
            ))
    finally:
        for writer in writers.values():
            writer.release()
    with (output_dir / f"{stem}_isolados.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{stem}_comparacao.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return {"input": str(input_path), "frames": len(frames), "janelas": len(comparisons)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/isolados")
    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)
    config.validate()
    device = resolve_device(config.device)
    detector = YoloPersonDetector(config.yolo, device)
    segmenter = build_segmenter(config.unet, device)
    classifier = SlowFastClassifier(config.slowfast, device)
    for path in args.inputs:
        print(run_individual(path, args.output, config, detector, segmenter, classifier))


if __name__ == "__main__":
    main()
