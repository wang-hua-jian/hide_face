"""P0 配置测试：单一参数源、校验错误、环境变量覆盖、模型自检。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from config.settings import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    Settings,
    ensure_models,
)


def _default_data() -> dict:
    return copy.deepcopy(
        yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_default_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIDEFACE_CONFIG", raising=False)
    monkeypatch.delenv("HIDEFACE_YUNET_PATH", raising=False)
    monkeypatch.delenv("HIDEFACE_SFACE_PATH", raising=False)

    settings = Settings.load()

    assert settings.detector.score_threshold == pytest.approx(0.7)
    assert settings.tracker.high_score_threshold == pytest.approx(0.7)
    assert settings.tracker.appearance_similarity == pytest.approx(0.4)
    assert settings.tracker.appearance_max_lost_frames == 5
    assert settings.tracker.eye_ratio_min_samples == 20
    assert settings.tracker.eye_ratio_lower_factor == pytest.approx(0.4)
    assert settings.tracker.eye_ratio_fallback == pytest.approx(0.12)
    assert settings.render.style == "solid"
    assert settings.render.mask_shape == "rect"
    assert settings.target.box_expand_ratio == pytest.approx(0.0)
    assert settings.camera.max_fps == 15
    assert settings.camera.display_scale == pytest.approx(1.0)
    assert Path(settings.models.yunet_path).is_absolute()
    assert Path(settings.models.sface_path).is_absolute()


def test_yaml_is_the_only_source(tmp_path: Path) -> None:
    data = _default_data()
    data["detector"]["score_threshold"] = 0.11

    settings = Settings.load(_write_config(tmp_path, data))

    assert settings.detector.score_threshold == pytest.approx(0.11)


def test_missing_field_reports_name(tmp_path: Path) -> None:
    data = _default_data()
    del data["detector"]["nms_threshold"]

    with pytest.raises(ConfigError) as excinfo:
        Settings.load(_write_config(tmp_path, data))

    assert "detector.nms_threshold" in str(excinfo.value)


def test_wrong_type_reports_field(tmp_path: Path) -> None:
    data = _default_data()
    data["detector"]["score_threshold"] = "high"

    with pytest.raises(ConfigError) as excinfo:
        Settings.load(_write_config(tmp_path, data))

    assert "detector.score_threshold" in str(excinfo.value)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    data = _default_data()
    data["detector"]["unexpected_key"] = 1

    with pytest.raises(ConfigError) as excinfo:
        Settings.load(_write_config(tmp_path, data))

    assert "detector.unexpected_key" in str(excinfo.value)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="配置文件不存在"):
        Settings.load(tmp_path / "not_exists.yaml")


def test_env_override_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIDEFACE_YUNET_PATH", "custom/yunet_custom.onnx")
    data = _default_data()

    settings = Settings.load(_write_config(tmp_path, data))

    yunet_path = Path(settings.models.yunet_path)
    assert yunet_path.name == "yunet_custom.onnx"
    assert yunet_path.is_absolute()


def test_ensure_models_missing(tmp_path: Path) -> None:
    data = _default_data()
    data["models"]["yunet_path"] = str(tmp_path / "missing_yunet.onnx")
    data["models"]["sface_path"] = str(tmp_path / "missing_sface.onnx")

    settings = Settings.load(_write_config(tmp_path, data))

    with pytest.raises(ConfigError) as excinfo:
        ensure_models(settings)
    assert "模型自检失败" in str(excinfo.value)
    assert "missing_yunet.onnx" in str(excinfo.value)


def test_ensure_models_empty_file(tmp_path: Path) -> None:
    empty_model = tmp_path / "empty.onnx"
    empty_model.write_bytes(b"")
    ok_model = tmp_path / "ok.onnx"
    ok_model.write_bytes(b"not-a-real-model")

    data = _default_data()
    data["models"]["yunet_path"] = str(empty_model)
    data["models"]["sface_path"] = str(ok_model)

    settings = Settings.load(_write_config(tmp_path, data))

    with pytest.raises(ConfigError) as excinfo:
        ensure_models(settings)
    assert "为空" in str(excinfo.value)


def test_ensure_models_passes_for_temp_files(tmp_path: Path) -> None:
    yunet = tmp_path / "yunet.onnx"
    sface = tmp_path / "sface.onnx"
    yunet.write_bytes(b"yunet")
    sface.write_bytes(b"sface")

    data = _default_data()
    data["models"]["yunet_path"] = str(yunet)
    data["models"]["sface_path"] = str(sface)

    settings = Settings.load(_write_config(tmp_path, data))

    ensure_models(settings)


def test_missing_sface_allowed_when_appearance_disabled(tmp_path: Path) -> None:
    yunet = tmp_path / "yunet.onnx"
    yunet.write_bytes(b"yunet")

    data = _default_data()
    data["models"]["yunet_path"] = str(yunet)
    data["models"]["sface_path"] = str(tmp_path / "missing_sface.onnx")
    data["tracker"]["appearance_enabled"] = False

    settings = Settings.load(_write_config(tmp_path, data))

    ensure_models(settings)
