import csv
import json

import cv2
import numpy as np
import pytest

from action_vision.config import PipelineConfig
from action_vision.pipeline import VideoPipeline
from action_vision.types import BoundingBox, Detection


class Detector:
    def detect(self, frame):
        return [Detection(BoundingBox(10, 12, 30, 36), 0.8)]


class Segmenter:
    def segment(self, crop):
        return np.ones(crop.shape[:2], dtype=bool)


class Classifier:
    def __init__(self):
        self.calls = 0

    def classify(self, frames):
        self.calls += 1
        return [(f"acao_{self.calls}", 0.9)]


@pytest.mark.parametrize("count", [32, 33, 64, 100])
def test_real_video_io_covers_every_frame_and_closes_tail(tmp_path, count):
    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    config = PipelineConfig()
    classifier = Classifier()
    result = VideoPipeline(config, Detector(), Segmenter(), classifier).run(source, tmp_path / "output")
    with open(result["csv"], encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with open(result["json"], encoding="utf-8") as stream:
        predictions = json.load(stream)
    assert len(rows) == count
    assert predictions[0]["frame_inicial"] == 0
    assert predictions[-1]["frame_final"] == count - 1
    assert all(row["acao"] for row in rows)
    assert all(int(row["area_mascara_px"]) > 0 for row in rows)
    for index, row in enumerate(rows):
        assert int(row["frame_inicial_janela"]) <= index <= int(row["frame_final_janela"])
    capture = cv2.VideoCapture(result["video"])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    capture.release()
    assert len(list((tmp_path / "output").glob("*.mp4"))) == 1


def test_pipeline_prefers_segmentable_person_over_confident_background(tmp_path):
    class MultiplePeople:
        def detect(self, frame):
            return [
                Detection(BoundingBox(2, 2, 20, 40), 0.65),
                Detection(BoundingBox(35, 2, 60, 50), 0.95),
            ]

    class BrightPerson:
        def segment(self, crop):
            return crop[:, :, 0] > 150

    source = tmp_path / "people.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[2:40, 2:20] = 255
    for _ in range(32):
        writer.write(frame)
    writer.release()
    result = VideoPipeline(PipelineConfig(), MultiplePeople(), BrightPerson(), Classifier()).run(source, tmp_path)
    with open(result["csv"], encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["bbox_x1"] == "2" for row in rows)
    assert all(row["candidatos_mascara_naovazia"] == "1" for row in rows)
