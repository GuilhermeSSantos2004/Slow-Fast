"""Extrai inicio, meio e fim de cada video anotado (processamento offline)."""

from __future__ import annotations

import argparse
from pathlib import Path


def extract(video_path: Path, output_dir: Path) -> list[Path]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes = sorted({max(0, round((total - 1) * fraction)) for fraction in (0.00, 0.50, 1.00)})
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for number, frame_index in enumerate(indexes, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        destination = output_dir / f"{video_path.stem}_momento_{number}.jpg"
        cv2.imwrite(str(destination), frame)
        created.append(destination)
    capture.release()
    return created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--output", default="docs/images/evidencias")
    args = parser.parse_args()
    for value in args.videos:
        for path in extract(Path(value), Path(args.output)):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
