from pathlib import Path

import pytest

from action_vision.config import PipelineConfig


def test_default_config_is_valid() -> None:
    config = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    config.validate()
    assert config.slowfast.window_size == 32
    assert config.unet.backend == "unet_human"


def test_invalid_stride_is_rejected() -> None:
    config = PipelineConfig()
    config.slowfast.stride = 0
    with pytest.raises(ValueError, match="stride"):
        config.validate()

