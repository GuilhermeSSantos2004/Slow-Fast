"""Adaptadores dos tres modelos pre-treinados exigidos no CP5."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.request import urlretrieve

import numpy as np

from .config import SlowFastConfig, UnetConfig, YoloConfig
from .types import BoundingBox, Detection


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class Segmenter(Protocol):
    def segment(self, crop_bgr: np.ndarray) -> np.ndarray: ...


class ActionClassifier(Protocol):
    def classify(self, frames_bgr: Sequence[np.ndarray]) -> list[tuple[str, float]]: ...


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class YoloPersonDetector:
    """YOLO pre-treinado em COCO, filtrando somente a classe pessoa (0)."""

    def __init__(self, config: YoloConfig, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Instale as dependencias com: pip install -r requirements.txt") from exc
        self.config = config
        self.device = device
        self.model = YOLO(config.weights)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            classes=[0],
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for coords, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = np.rint(coords).astype(int).tolist()
            detections.append(Detection(BoundingBox(x1, y1, x2, y2), float(confidence)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


class U2NetHumanSegmenter:
    """U2-Net pre-treinada para segmentacao humana, uma U-Net aninhada."""

    def __init__(self, config: UnetConfig) -> None:
        try:
            from rembg import new_session
        except ImportError as exc:
            raise RuntimeError("O backend U2-Net requer o pacote rembg.") from exc
        self.config = config
        self.session = new_session(config.model_name)

    def segment(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        from rembg import remove

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mask = remove(crop_rgb, session=self.session, only_mask=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask.astype(np.float32) / 255.0) >= self.config.threshold


class UnetHumanSegmenter:
    """U-Net completa pre-treinada para pessoas; nenhum ajuste de pesos."""

    SOURCE_REVISION = "d5f9ab4afd5e0f9aedaa7c1565d506e8e650f916"
    WEIGHTS_URL = (
        "https://raw.githubusercontent.com/amangupta143/PyTorch-Image-Segmentation/"
        f"{SOURCE_REVISION}/Trained%20Model/bestModel.pt"
    )
    WEIGHTS_SHA256 = "cec0c202bad87f45a6da2c864989c22b44ece24ffe5819d8ab726806ae43e6f5"

    def __init__(self, config: UnetConfig, device: str) -> None:
        import segmentation_models_pytorch as smp
        import torch

        self.config = config
        self.device = device
        self.torch = torch
        checkpoint = Path(config.checkpoint) if config.checkpoint else Path("models/unet_human.pt")
        if not checkpoint.exists():
            if config.checkpoint:
                raise FileNotFoundError(f"Checkpoint U-Net nao encontrado: {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".download")
            try:
                urlretrieve(self.WEIGHTS_URL, temporary)
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if digest != self.WEIGHTS_SHA256:
                    raise RuntimeError("Checksum invalido para o checkpoint U-Net.")
                temporary.replace(checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
        if not config.checkpoint and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.WEIGHTS_SHA256:
            raise RuntimeError("O checkpoint U-Net em cache nao corresponde aos pesos publicados.")
        # encoder_weights=None evita baixar um encoder e deixa claro que o estado
        # completo (encoder + decoder) e restaurado, com validacao estrita.
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b0", encoder_weights=None,
            in_channels=3, classes=1, activation=None,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("arc."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def segment(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.config.input_size, self.config.input_size))
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with self.torch.inference_mode():
            logits = self.model(tensor.to(self.device))
            probability = logits.sigmoid()[0, 0].cpu().numpy()
        return probability >= self.config.threshold


class TorchScriptUnetSegmenter:
    """Adaptador para o checkpoint U-Net fornecido na disciplina."""

    def __init__(self, config: UnetConfig, device: str) -> None:
        if not config.checkpoint:
            raise ValueError("Informe unet.checkpoint para usar backend=torchscript.")
        checkpoint = Path(config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint U-Net nao encontrado: {checkpoint}")
        import torch

        self.torch = torch
        self.device = device
        self.threshold = config.threshold
        self.input_size = config.input_size
        self.model = torch.jit.load(str(checkpoint), map_location=device).eval()

    def segment(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        tensor = tensor.to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probability = output.sigmoid().squeeze().detach().cpu().numpy()
        return probability >= self.threshold


class SlowFastClassifier:
    """SlowFast R50 pre-treinado no Kinetics-400."""

    MEAN = (0.45, 0.45, 0.45)
    STD = (0.225, 0.225, 0.225)

    def __init__(self, config: SlowFastConfig, device: str) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.device = device
        labels_path = Path(config.labels)
        self.labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(self.labels) != 400:
            raise ValueError(f"Esperadas 400 classes Kinetics; recebidas {len(self.labels)}.")
        # Pacote versionado, sem executar codigo mutavel de um branch via hub.
        from pytorchvideo.models.hub import slowfast_r50

        if config.model != "slowfast_r50":
            raise ValueError("Este projeto suporta slowfast_r50 pre-treinado.")
        self.model = slowfast_r50(pretrained=True).eval().to(device)

    def _prepare(self, frames_bgr: Sequence[np.ndarray]):
        import cv2

        resized = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            scale = self.config.crop_size / min(height, width)
            new_size = (round(width * scale), round(height * scale))
            image = cv2.resize(rgb, new_size, interpolation=cv2.INTER_LINEAR)
            y0 = max(0, (image.shape[0] - self.config.crop_size) // 2)
            x0 = max(0, (image.shape[1] - self.config.crop_size) // 2)
            image = image[y0 : y0 + self.config.crop_size, x0 : x0 + self.config.crop_size]
            resized.append(image)
        array = np.stack(resized).astype(np.float32) / 255.0
        video = self.torch.from_numpy(array).permute(3, 0, 1, 2)
        mean = self.torch.tensor(self.MEAN).view(3, 1, 1, 1)
        std = self.torch.tensor(self.STD).view(3, 1, 1, 1)
        fast = ((video - mean) / std).unsqueeze(0).to(self.device)
        slow_count = max(1, fast.shape[2] // self.config.alpha)
        indices = self.torch.linspace(0, fast.shape[2] - 1, slow_count).long().to(self.device)
        slow = self.torch.index_select(fast, 2, indices)
        return [slow, fast]

    def classify(self, frames_bgr: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        inputs = self._prepare(frames_bgr)
        with self.torch.inference_mode():
            probabilities = self.model(inputs).softmax(dim=1)[0]
        scores, indexes = probabilities.topk(self.config.top_k)
        return [(self.labels[int(index)], float(score)) for score, index in zip(scores.cpu(), indexes.cpu())]


def build_segmenter(config: UnetConfig, device: str) -> Segmenter:
    if config.backend == "unet_human":
        return UnetHumanSegmenter(config, device)
    if config.backend == "u2net_human":
        return U2NetHumanSegmenter(config)
    if config.backend == "torchscript":
        return TorchScriptUnetSegmenter(config, device)
    raise ValueError(f"Backend U-Net desconhecido: {config.backend}")
