# M0-03 环境检查报告

检查时间：2026-10-02T11:44:08.074569+08:00。
任务验收：**未通过或未完成**。
原始证据：[本次 JSON 快照](../logs/environment/M0-03_20261002_114408_074569_snapshot.json)。

## 1. 解释器与训练环境

| 检查 | 结果 |
| --- | --- |
| Python | 3.9.23 |
| 解释器 | `D:\software\anaconda\envs\yolo\python.exe` |
| 训练环境检查 | passed |
| Ultralytics | 8.4.171，editable=True |
| 源码 commit | `86f5c8c401a630da651d321e691ab0af540812cf` |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| torch / CUDA | 2.8.0+cu129 / 12.9 |
| 依赖检查 | No broken requirements found. |
| 模型运行 | 随机权重 YOLO11n-seg，640×640，FP32，GPU 输出有限 |
| OpenCV | PNG 编解码后像素一致 |

## 2. 摄像头

状态：**failed**；设备索引：0。

- DSHOW：失败。
  错误：Timeout after 15.0s
  超时前读到 3 帧，形状 [480, 640, 3]；设备释放完成：False。

```text
stage=opening
stage=opened success=True
stage=reading attempt=1
stage=reading attempt=2
stage=reading attempt=3
camera_result={"index": 0, "requested_backend": "DSHOW", "mode": "native", "passed": true, "valid_frames": 3, "released": false, "opencv_version": "4.11.0", "opencv_file": "D:\\software\\anaconda\\envs\\yolo\\lib\\site-packages\\cv2\\__init__.py", "opened": true, "actual_backend": "DSHOW", "frame_shape": [480, 640, 3], "frame_dtype": "uint8", "frame_min": 0, "frame_max": 248, "reported_fps": 0.0}
stage=releasing
```


检查尝试读取少量帧并释放设备；驱动卡住时终止该检查的子进程。没有保存或显示画面。设备报告帧率不是实测应用 FPS。

## 3. 部署环境状态

| 模块 | 当前状态 | 后续 |
| --- | --- | --- |
| onnx | not_installed | M0-05 安装/执行验证 |
| onnxruntime | not_installed | M0-05 安装/执行验证 |
| tensorrt | not_installed | M0-05 安装/执行验证 |

部署依赖缺失在本阶段如实登记，不作为训练环境或摄像头检查已通过的证据。provider 列表也不等同于模型实际执行。

## 4. 验收边界与下一步

M0-03 验收要求：训练检查通过、摄像头实际读到三帧并正常释放，并记录部署模块状态。
尚未验证：E0 预训练模型、训练/反向、ONNX 模型执行、TensorRT 构建/执行和长时间摄像头稳定性。
下一步：解决报告中的未通过项并重新检查，再验收 M0-03。
