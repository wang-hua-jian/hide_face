"""配置加载与校验。

唯一参数来源是 config/default.yaml；本模块不保存任何阈值默认值。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
MASKS_DIR = DATA_DIR / "masks"
REFERENCES_DIR = DATA_DIR / "references"
PROFILES_DIR = DATA_DIR / "profiles"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
OUTPUT_JSON_DIR = OUTPUT_DIR / "json"
OUTPUT_LIVE_DIR = OUTPUT_DIR / "live"
OUTPUT_DEBUG_DIR = OUTPUT_DIR / "debug"

RUNTIME_DIRS = (
    IMAGES_DIR,
    VIDEOS_DIR,
    MASKS_DIR,
    REFERENCES_DIR,
    PROFILES_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_VIDEOS_DIR,
    OUTPUT_JSON_DIR,
    OUTPUT_LIVE_DIR,
    OUTPUT_DEBUG_DIR,
)


class ConfigError(RuntimeError):
    """配置文件缺失、格式错误或模型自检失败。"""


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yunet_path: str
    sface_path: str


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_threshold: float = Field(..., ge=0.0, le=1.0)
    nms_threshold: float = Field(..., ge=0.0, le=1.0)
    max_input_size: int = Field(..., gt=0)
    detect_interval: int = Field(..., ge=1)


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_score_threshold: float = Field(..., ge=0.0, le=1.0)
    low_score_threshold: float = Field(..., ge=0.0, le=1.0)
    iou_threshold: float = Field(..., ge=0.0, le=1.0)
    second_iou_threshold: float = Field(..., ge=0.0, le=1.0)
    max_lost_frames: int = Field(..., ge=0)
    mask_lost_frames: int = Field(..., ge=0)
    lost_mask_distance_ratio: float = Field(..., gt=0.0)
    smooth_alpha: float = Field(..., gt=0.0, le=1.0)
    max_scale_step: float = Field(..., ge=0.0, le=1.0)
    appearance_similarity: float = Field(..., ge=-1.0, le=1.0)
    ema_alpha: float = Field(..., ge=0.0, le=1.0)


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 正值为向外扩张，负值为向内收缩；-0.5 会把区域缩到 0
    box_expand_ratio: float = Field(..., gt=-0.5, le=1.0)


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarity_threshold: float = Field(..., ge=-1.0, le=1.0)
    max_embeddings: int = Field(..., gt=0)
    profile_dir: str


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: Literal["mosaic", "blur", "solid", "custom_image"]
    mask_shape: Literal["rect", "ellipse"]
    opacity: float = Field(..., ge=0.0, le=1.0)
    mosaic_block: int = Field(..., ge=2)
    blur_kernel: int = Field(..., ge=3)
    jpeg_quality: int = Field(..., ge=1, le=100)


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0)
    max_fps: int = Field(..., gt=0)
    display_scale: float = Field(..., gt=0.0, le=1.0)
    snap_count: int = Field(..., ge=1)
    snap_interval_seconds: float = Field(..., ge=0.0)


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_seconds: int = Field(..., gt=0)
    chunk_overlap_seconds: int = Field(..., ge=0)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: ModelsConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    target: TargetConfig
    identity: IdentityConfig
    render: RenderConfig
    camera: CameraConfig
    video: VideoConfig

    def model_post_init(self, __context: object) -> None:
        self.models.yunet_path = _resolve_project_path(self.models.yunet_path)
        self.models.sface_path = _resolve_project_path(self.models.sface_path)
        self.identity.profile_dir = _resolve_project_path(self.identity.profile_dir)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        config_path = Path(path) if path is not None else _config_path_from_env()
        if not config_path.is_file():
            raise ConfigError(f"配置文件不存在：{config_path}")

        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"配置文件格式错误：{config_path}\n{exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件内容必须是映射结构：{config_path}")

        raw = _apply_env_overrides(raw)
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc)) from exc


def ensure_models(settings: Settings) -> None:
    """启动自检：两个 onnx 必须存在且非空。"""

    problems = []
    for label, value in (
        ("YuNet", settings.models.yunet_path),
        ("SFace", settings.models.sface_path),
    ):
        model_path = Path(value)
        if not model_path.is_file():
            problems.append(f"{label} 模型缺失：{model_path}")
        elif model_path.stat().st_size == 0:
            problems.append(f"{label} 模型文件为空：{model_path}")

    if problems:
        detail = "\n".join(f"  - {item}" for item in problems)
        raise ConfigError(f"模型自检失败，请将 onnx 权重放入 models/：\n{detail}")


def ensure_runtime_dirs() -> None:
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def _config_path_from_env() -> Path:
    override = os.environ.get("HIDEFACE_CONFIG")
    return Path(override) if override else DEFAULT_CONFIG_PATH


def _apply_env_overrides(raw: dict) -> dict:
    models = raw.get("models")
    if isinstance(models, dict):
        yunet_override = os.environ.get("HIDEFACE_YUNET_PATH")
        sface_override = os.environ.get("HIDEFACE_SFACE_PATH")
        if yunet_override:
            models["yunet_path"] = yunet_override
        if sface_override:
            models["sface_path"] = sface_override
    return raw


def _resolve_project_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["配置校验失败："]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
