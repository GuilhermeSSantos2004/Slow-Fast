"""Tipos compartilhados pelo pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Caixa no formato de cantos, em pixels."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip(self, frame_width: int, frame_height: int) -> BoundingBox:
        """Limita a caixa aos limites validos do quadro."""

        return BoundingBox(
            x1=max(0, min(self.x1, frame_width)),
            y1=max(0, min(self.y1, frame_height)),
            x2=max(0, min(self.x2, frame_width)),
            y2=max(0, min(self.y2, frame_height)),
        )

    def expand(self, ratio: float, frame_width: int, frame_height: int) -> BoundingBox:
        """Expande a caixa mantendo o centro e respeitando o quadro."""

        dx = round(self.width * ratio / 2)
        dy = round(self.height * ratio / 2)
        return BoundingBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clip(
            frame_width, frame_height
        )


@dataclass(slots=True)
class Detection:
    """Deteccao de uma pessoa e sua mascara opcional."""

    box: BoundingBox
    confidence: float
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    """Acao estimada para uma janela temporal."""

    label: str
    confidence: float
    start_frame: int
    end_frame: int
    ranked: tuple[tuple[str, float], ...] = ()
