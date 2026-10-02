# YOLO 学习与项目入口

更新日期：2026-10-02。

从 [YOLO 学习路线与学习手册（终极版）](codex/00_YOLO学习路线与学习手册_终极版.md) 开始。主线为检测基础 → 现代检测 → 训练机制 → 实例分割与源码 → 基线与错误分析，共六阶段十八课；YOLOv10、YOLO26 和部署为进阶方向。

- [YOLOv1 原论文精读](codex/YOLOv1_原论文精读与基础模型学习.md)：基础卷的详细讲解。
- [现代检测与实例分割过渡讲义](codex/02_从YOLOv1到现代YOLO_检测与实例分割基础.md)：多尺度、分配、DFL、原型掩膜等概念。
- [原始论文、版本与来源](ref/README_文献目录与来源.md)：查阅文献和固定源码快照。
- [工程进度与任务验收](plan/00_YOLO_分阶段推进与进度跟踪.md)：当前三类 YOLO11n-seg 项目的环境、数据、实验与部署。

建议首次完成学习手册第 1 课的任务区分、坐标与 IoU 练习。资料准备、知识掌握、实际运行和实验有效性分别验收。

资料维护脚本位于 [scripts/download_learning_refs.py](scripts/download_learning_refs.py)。`tmp/` 用于可重建的文本提取和页面查看缓存，可在不用时清理，已加入 Git 忽略规则；原始论文与来源记录保存在 `ref/`。

## 工程环境与当前进展

M0-01 至 M0-04 已完成，主线已验收 **4/54**。专用 `yolo` 环境、Ultralytics 8.4.171 源码锁定和正式环境检查均通过；YOLO11n-seg 的 E0 预训练权重已在 GPU 上运行图片、视频与摄像头，保存输入、检测框和实例掩膜。10 秒摄像头演示处理 297 帧并正常关闭；识别到右侧手机与杯子，左侧遮挡手机漏检、右侧瓶子出现漏检 / 误分类，已如实记录。下一步 M0-05 检查 ONNX / TensorRT 导出与执行兼容性。

- [环境安装记录](docs/环境安装记录.md)：版本、实际命令、问题处理与恢复顺序。
- [训练依赖](requirements-train.txt)、[源码锁定清单](configs/source-lock.json)、[统一路径配置](configs/paths.yaml)。
- [环境报告](reports/environment.md)：硬件与本次运行证据。
- [正式环境检查报告](reports/environment_check.md)、[检查使用说明](docs/环境检查使用说明.md)：一键检查与当前摄像头问题。
- [E0 使用说明](docs/E0_预训练模型使用说明.md)、[运行与效果报告](reports/E0_预训练模型验证.md)、[推理配置](configs/e0.yaml)：图片、视频与摄像头演示。

本机从项目根目录复查：

```powershell
conda activate yolo
python scripts/check_env.py
```

解释器应为 `D:\software\anaconda\envs\yolo\python.exe`。移机时先按安装记录恢复源码和依赖；源码恢复入口为 `python scripts/setup_source.py`。当前训练依赖已锁定，部署依赖在 M0-05 验证后另行记录。

运行 E0：

```powershell
python scripts/prepare_e0.py
python scripts/predict_e0.py --source demo/input/E0/bus.jpg --all-classes
python scripts/predict_e0.py --camera --seconds 10 --show
```

默认保存到 `runs/E0/<时间戳>/`；摄像头输入与结果视频都会保存，显示窗口按 Q / Esc 或到期退出。原模型保留 COCO 80 类，默认通过类别名称筛选杯子、瓶子、手机；这一步未进行项目微调或正式精度评价。
