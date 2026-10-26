# 模型权重

本目录存放运行时所需的 ONNX 权重，权重文件不入 git（见 `.gitignore` 中的 `models/*.onnx`）。

## 需要的文件

| 文件名 | 来源 | 说明 |
|---|---|---|
| `yunet.onnx` | OpenCV Zoo `face_detection_yunet_2023mar.onnx` | 人脸检测 + 5 点关键点 |
| `sface.onnx` | OpenCV Zoo `face_recognition_sface_2021dec.onnx` | 人脸特征（身份增强 / 外观兜底） |

下载地址：

```text
YuNet   https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
SFace   https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
```

下载后重命名为上表文件名，放入本目录即可。

## 校验

```bash
python scripts/check_models.py
```

脚本会读取 `config/default.yaml` 中的模型路径，检查文件存在且非空，并实际创建 YuNet / SFace 实例。

## 路径覆盖

默认从 `config/default.yaml` 的 `models.yunet_path` / `models.sface_path` 读取，也可用环境变量覆盖：

```text
HIDEFACE_YUNET_PATH
HIDEFACE_SFACE_PATH
```
