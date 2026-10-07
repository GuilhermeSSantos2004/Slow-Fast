"""Baixa os exemplos publicos e prepara os mesmos trechos, sem treinar modelos."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    {
        "id": "video_1_archery", "file": "archery.mp4", "start_s": 0, "duration_s": 10,
        "size": "320:240", "sha256": "8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4",
        "reference": "https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/torchhub_inference_tutorial.ipynb",
    },
    {
        "id": "video_2_theatre", "file": "theatre.webm", "start_s": 1, "duration_s": 10,
        "size": "320:180", "sha256": "1830e4f0c4ddf5de96695d3df5b1da80bf6c6e26382c6ced01a5fbd0ffde8612",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/projects/theatre.webm",
        "reference": "https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_detection_example/video_detection_inference_tutorial.ipynb",
    },
]


def prepare() -> None:
    cache = ROOT / "tmp" / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "assets" / "videos"
    destination.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        original = cache / source["file"]
        if not original.exists():
            urlretrieve(source["url"], original)
        if hashlib.sha256(original.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"A fonte mudou: {source['file']}; confira antes de reproduzir.")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", str(source["start_s"]), "-i", str(original),
            "-t", str(source["duration_s"]), "-vf", f"fps=8,scale={source['size']}",
            "-an", "-c:v", "libx264", "-crf", "18", str(destination / f"{source['id']}.mp4"),
        ], check=True)


if __name__ == "__main__":
    prepare()
