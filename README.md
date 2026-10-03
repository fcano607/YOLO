# YOLO 学习与项目入口

更新：2026-10-03。项目主线：杯子、瓶子、手机的 YOLO11n-seg 实例分割、场景适配与独立部署。

**M0 环境模块已完成：6/6 项；主线 6/54 项，完整模块 1/9。** 当前是官方 COCO 80 类预训练模型，默认筛选三类；已有图片 / 视频 / 摄像头、ONNX、ORT CPU/CUDA、TensorRT FP32 原始输出和新进程复现证据。项目训练、结构改进、完整独立后处理与正式精度 / 速度评价尚未开展。

**M1 按项目与面试目标推进：**工程证据已齐，个人理解仍待核对，M1 保持 0/6。2026-10-03 按用户目标改为直接讲解、实际例子、少量追问；当前推进 M1-02 整体结构，入口为[项目与面试学习方式](docs/模型学习笔记.md#interview-route)。原详细笔记与终极版手册作为按需查阅资料，核心流程明确后衔接数据准备与真实训练。

## 常用入口

| 想查看什么 | 文档 |
| --- | --- |
| 当前进度与下一步 | [分阶段任务与进度跟踪](plan/00_YOLO_分阶段推进与进度跟踪.md) |
| 如何运行、检查和恢复 | [M0 环境与模型使用手册](docs/M0_环境与模型使用手册.md) |
| 实时摄像头双画面预览 | [Live Camera：启动、按键与类别筛选源码](docs/M0_环境与模型使用手册.md#live-camera) |
| 已完成结果与验收证据 | [M0 阶段验收报告](reports/M0_阶段验收报告.md) |
| 模型原理学习 | [六阶段十八课学习手册](codex/00_YOLO学习路线与学习手册_终极版.md) |
| 当前 M1 怎么学、实测了什么 | [项目与面试入口](docs/模型学习笔记.md#interview-route)、[锁定模型实际观察报告](reports/model_notes.md) |
| 各版本改了什么 | [YOLO 各版本演进与主要改动速览](codex/03_YOLO各版本演进与主要改动速览.md) |
| 第一次从哪里学 | [第一课：从一张图理解目标检测](codex/01_第一课_从一张图理解目标检测.md)，跟读、手算，再完成五道练习 |
| 文件职责与清理结果 | [文件整理与清理清单](reports/maintenance/20261002_文件整理与清理清单.md) |

## 本机运行

```powershell
Set-Location 'E:\秋招\项目相关\YOLO'
conda activate yolo
python -c "import sys; print(sys.executable)"
python scripts/prepare_e0.py
python scripts/predict_e0.py --source demo/input/E0/bus.jpg --all-classes
```

解释器应为 `D:\software\anaconda\envs\yolo\python.exe`。摄像头命令会保存原始画面与结果，窗口按 Q / Esc 或到期退出：

```powershell
python scripts/predict_e0.py --camera --seconds 10 --show
```

环境检查、部署预检查、环境归档与复现命令均集中在使用手册。当前不需要重复安装依赖。正式实验保存新的输出目录，避免覆盖已有证据。

实时查看原图与分割结果，默认不录制：

```powershell
python app/live_camera.py
```

默认左侧原图、右侧 YOLO 结果。窗口内按 1/2/3 切换视图，Q / Esc 或关闭窗口退出；默认持续运行。加 `--all-classes` 可取消三类筛选，仍使用同一个预训练模型。详见[双画面使用说明](docs/M0_环境与模型使用手册.md#live-camera)。该 M7 原型已做 GPU 图片和模拟交互检查，真实摄像头窗口及长期稳定性尚待验收；本次操作已登记在[工作日志](plan/00_YOLO_分阶段推进与进度跟踪.md#live-camera-record)。

## 目录职责

| 类别 | 目录 / 文件 | 用途 |
| --- | --- | --- |
| 计划与学习 | `plan/`、`codex/`、`ref/` | 活任务清单、原理讲义、原始论文 / 来源 |
| 可运行工程 | `scripts/`、`deploy/`、`configs/`、`app/` | 工程入口、共享代码、配置与实时预览原型 |
| 数据与模型 | `data/`、`artifacts/` | 数据 / 标签 / 固定样本、预训练权重及部署文件 |
| 运行与验收 | `runs/`、`demo/`、`logs/`、`reports/` | 原始运行产物、演示、机器记录、人工结论 |
| 使用与恢复 | `docs/`、根目录 requirements / environment.yml | 合并操作说明与环境恢复定义 |
| 外部源码与缓存 | `third_party/`、`tmp/` | 锁定上游源码、可重建缓存 |
| Git 内部目录 | `.git/` | 版本历史与仓库数据，保留 |

目录路径是现有代码 / 配置的一部分，主要通过合并说明、归档历史日志和清理重复产物减少分散文件。数据和模型目录保留原位。`data/fixed_checks/` 等必要空目录属于路径配置，继续保留。

## 深入资料与历史

- [完整实施方案](plan/YOLO_实时实例分割项目_完整实施方案.md)与[目录规划](plan/YOLO_项目目录规划.md)：后续模块设计，规划中的文件不代表已经实现。
- [YOLOv1 精读](codex/YOLOv1_原论文精读与基础模型学习.md)、[现代检测与实例分割讲义](codex/02_从YOLOv1到现代YOLO_检测与实例分割基础.md)：保留内容不同的学习材料。
- [论文与来源](ref/README_文献目录与来源.md)、[资料维护脚本](scripts/download_learning_refs.py)：原始论文保留，tmp 中提取文本 / 页面属于缓存。
- [源码锁定](configs/source-lock.json)、[训练需求](requirements-train.txt)、[部署需求](requirements-deploy.txt)、[Conda 定义](environment.yml)、[当前版本约束](configs/constraints-runtime.txt)：恢复工程所需。
- [M0 历史原文备份](logs/archive/M0_历史记录_20261002.zip)：整理前日志、旧说明 / 报告和诊断证据，内置逐文件 SHA256 索引。

此前已按用户同意的推荐清单删除 301 个文件、132 个目录，移除约 2.31 GiB 缓存和冗余产物。217 个历史文件的压缩备份保留；GPU 图片推理、ORT CPU/CUDA 和 TensorRT 运行复查通过。具体结果见清理清单。当前按面试目标推进 M1-02，M1-01 基础已讨论，个人完整复述仍待核对。
