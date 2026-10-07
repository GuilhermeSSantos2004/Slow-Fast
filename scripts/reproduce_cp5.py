"""Reproduz inferencia, modelos isolados, comparacao, evidencias e PDF."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from prepare_videos import SOURCES, prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-sources", action="store_true", help="Recria os trechos a partir dos originais.")
    args = parser.parse_args()
    if args.download_sources:
        prepare()
    inputs = [f"assets/videos/{source['id']}.mp4" for source in SOURCES]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env["PYTHONPATH"] = str(ROOT / "src")
    commands = [
        ["-m", "action_vision", *inputs, "--output", "docs/results"],
        ["scripts/run_individual.py", *inputs, "--output", "docs/results/isolados"],
        ["scripts/extract_evidence.py", *[f"docs/results/{source['id']}_anotado.mp4" for source in SOURCES]],
        ["scripts/build_report.py"],
        ["scripts/build_demo.py"],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    packages = ["torch", "torchvision", "ultralytics", "segmentation-models-pytorch", "pytorchvideo", "numpy"]
    manifest = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "source_videos": SOURCES,
        "input_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in inputs},
        "config_sha256": hashlib.sha256((ROOT / "configs/default.yaml").read_bytes()).hexdigest(),
        "commands": [["python", *command] for command in commands],
        "threads": {"OMP_NUM_THREADS": env["OMP_NUM_THREADS"], "MKL_NUM_THREADS": env["MKL_NUM_THREADS"]},
    }
    (ROOT / "docs/results/reproducao.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
