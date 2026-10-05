"""Interface de linha de comando do projeto."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .models import SlowFastClassifier, YoloPersonDetector, build_segmenter, resolve_device
from .pipeline import VideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="action-vision",
        description="Integra YOLO, U-Net e SlowFast em videos curtos.",
    )
    parser.add_argument("inputs", nargs="+", help="Um ou mais videos de entrada.")
    parser.add_argument("--config", default="configs/default.yaml", help="Arquivo YAML de configuracao.")
    parser.add_argument("--output", default="outputs", help="Pasta de resultados.")
    parser.add_argument(
        "--person-crop",
        action="store_true",
        help="Executa SlowFast apenas no recorte da pessoa (extensao opcional).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PipelineConfig.from_yaml(args.config)
    if args.person_crop:
        config.slowfast.person_crop = True
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    print(f"Dispositivo selecionado: {device}")
    pipeline = VideoPipeline(
        config=config,
        detector=YoloPersonDetector(config.yolo, device),
        segmenter=build_segmenter(config.unet, device),
        classifier=SlowFastClassifier(config.slowfast, device),
    )
    summaries = []
    for input_value in args.inputs:
        summaries.append(pipeline.run(Path(input_value), Path(args.output)))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

