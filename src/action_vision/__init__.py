"""Deteccao, segmentacao e reconhecimento temporal de acoes humanas."""

from .config import PipelineConfig
from .types import ActionPrediction, BoundingBox, Detection

__all__ = ["ActionPrediction", "BoundingBox", "Detection", "PipelineConfig"]

