"""Configuracao validada do pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class YoloConfig:
    weights: str = "yolo11n.pt"
    confidence: float = 0.35
    iou: float = 0.45
    image_size: int = 640


@dataclass(slots=True)
class UnetConfig:
    backend: str = "unet_human"
    model_name: str = "u2net_human_seg"
    checkpoint: str | None = None
    input_size: int = 320
    threshold: float = 0.50
    box_margin: float = 0.10


@dataclass(slots=True)
class SlowFastConfig:
    model: str = "slowfast_r50"
    labels: str = "assets/labels/kinetics400.txt"
    window_size: int = 32
    stride: int = 16
    alpha: int = 4
    crop_size: int = 256
    top_k: int = 3
    person_crop: bool = False


@dataclass(slots=True)
class OutputConfig:
    mask_alpha: float = 0.40
    mask_color_bgr: list[int] = field(default_factory=lambda: [60, 180, 75])
    box_color_bgr: list[int] = field(default_factory=lambda: [0, 215, 255])
    codec: str = "mp4v"


@dataclass(slots=True)
class PipelineConfig:
    yolo: YoloConfig = field(default_factory=YoloConfig)
    unet: UnetConfig = field(default_factory=UnetConfig)
    slowfast: SlowFastConfig = field(default_factory=SlowFastConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            yolo=YoloConfig(**raw.get("yolo", {})),
            unet=UnetConfig(**raw.get("unet", {})),
            slowfast=SlowFastConfig(**raw.get("slowfast", {})),
            output=OutputConfig(**raw.get("output", {})),
            device=raw.get("device", "auto"),
            seed=int(raw.get("seed", 42)),
        )

    def validate(self) -> None:
        if not 0.0 < self.yolo.confidence <= 1.0:
            raise ValueError("yolo.confidence deve estar em (0, 1].")
        if not 0.0 < self.unet.threshold < 1.0:
            raise ValueError("unet.threshold deve estar em (0, 1).")
        if self.unet.backend not in {"unet_human", "torchscript", "u2net_human"}:
            raise ValueError("Backend U-Net desconhecido.")
        if self.unet.input_size <= 0 or self.unet.input_size % 32:
            raise ValueError("unet.input_size deve ser positivo e multiplo de 32.")
        if self.slowfast.alpha <= 0:
            raise ValueError("slowfast.alpha deve ser positivo.")
        if self.slowfast.window_size < self.slowfast.alpha:
            raise ValueError("window_size deve ser maior ou igual a alpha.")
        if self.slowfast.window_size != 32 or self.slowfast.alpha != 4:
            raise ValueError("O SlowFast R50 pre-treinado desta entrega usa window_size=32 e alpha=4.")
        if self.slowfast.stride <= 0:
            raise ValueError("slowfast.stride deve ser positivo.")
        if self.slowfast.stride > self.slowfast.window_size:
            raise ValueError("slowfast.stride nao pode deixar lacunas entre janelas.")
        if not 1 <= self.slowfast.top_k <= 400:
            raise ValueError("slowfast.top_k deve estar entre 1 e 400.")
        if len(self.output.codec) != 4:
            raise ValueError("output.codec deve ter quatro caracteres.")
