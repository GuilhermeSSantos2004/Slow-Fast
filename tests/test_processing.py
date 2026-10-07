import numpy as np

from action_vision.processing import crop_frames, project_mask, select_primary_person, temporal_windows
from action_vision.types import BoundingBox, Detection


def test_selection_keeps_overlapping_target_when_larger_person_enters():
    previous = BoundingBox(0, 0, 10, 10)
    target = Detection(BoundingBox(1, 0, 11, 10), 0.80)
    newcomer = Detection(BoundingBox(20, 0, 70, 50), 0.95)
    assert select_primary_person([target, newcomer], previous) is target


def test_project_mask_returns_full_frame_coordinates() -> None:
    crop_mask = np.ones((2, 2), dtype=bool)
    box = BoundingBox(2, 1, 6, 5)
    projected = project_mask(crop_mask, box, (8, 10, 3))
    assert projected.shape == (8, 10)
    assert int(projected.sum()) == 16
    assert projected[1:5, 2:6].all()


def test_primary_person_balances_area_and_confidence() -> None:
    detections = [
        Detection(BoundingBox(0, 0, 10, 10), 0.95),
        Detection(BoundingBox(0, 0, 50, 50), 0.70),
    ]
    assert select_primary_person(detections) == detections[1]


def test_temporal_windows_cover_last_frame() -> None:
    windows = temporal_windows(total_frames=100, window_size=32, stride=16)
    assert windows[0] == (0, 32)
    assert windows[-1] == (68, 100)


def test_crop_frames_uses_same_union_for_whole_clip() -> None:
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    boxes = [BoundingBox(5, 4, 12, 15), BoundingBox(8, 3, 18, 16)]
    crops = crop_frames(frames, boxes)
    assert crops[0].shape == crops[1].shape
    assert crops[0].shape[0] < 20
    assert crops[0].shape[1] < 30

