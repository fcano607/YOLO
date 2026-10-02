# YOLO 摄像头实时实例分割项目：目录规划

项目根目录：`E:\秋招\项目相关\YOLO`

本文件规划需要在你的 Windows 电脑中建立的目录与文件。2026-10-02 已完成 M0-02：必要一级目录、`configs/paths.yaml`、源码锁定清单、训练 requirements、路径解析与安装验收脚本已建立；后续数据、训练和部署脚本按任务逐步实现，表中其余文件仍是规划。实际进展见[进度跟踪](00_YOLO_分阶段推进与进度跟踪.md)。

同日已实现 `scripts/check_env.py`，生成 JSON 快照与 `reports/environment_check.md`；M0-03 的摄像头正常释放仍未通过，见[检查使用说明](../docs/环境检查使用说明.md)。

以下路径均相对于项目根目录，使用 `/` 表示子目录。例如 `configs/runtime.yaml` 在你的电脑上对应 `E:\秋招\项目相关\YOLO\configs\runtime.yaml`。

## 一、一级目录

| 目录 | 内容 | 创建时机 |
| --- | --- | --- |
| `configs/` | 数据集、训练、模型结构和推理参数配置 | 开始时 |
| `data/` | 原始资料、训练数据、标签、划分与固定推理样本 | 开始时 |
| `scripts/` | 环境检查、数据准备、训练、评估、导出与测速入口 | 开始时 |
| `deploy/` | 共享预处理、后处理、三种推理后端 | 开始时，可暂为空 |
| `app/` | 摄像头实时程序、采集、绘图与录制 | 开始时，可暂为空 |
| `third_party/` | 锁定版本的 Ultralytics 源码 | 准备环境时 |
| `artifacts/` | 下载的预训练模型，以及整理后的项目模型与部署文件 | 开始时 |
| `runs/` | 各次训练、评估、导出和测速产生的完整实验结果 | 实验执行时 |
| `reports/` | 数据统计、对照结果、部署一致性与性能分析 | 数据准备开始时 |
| `logs/` | 环境检查、构建、摄像头运行的文本日志 | 程序执行时 |
| `docs/` | 实施方案、目录规划、模型笔记与面试说明 | 开始时 |
| `demo/` | 摄像头截图、结果视频和固定测速视频 | 演示与部署阶段 |

这是完整项目结构，但不需要先把所有目录都填满。先完成环境检查，再按数据、训练、部署、实时应用逐步加入文件。

## 二、根目录文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | 项目介绍、环境安装、数据准备、训练、部署与运行命令 |
| `requirements-train.txt` | 训练环境的 Python 依赖与版本 |
| `requirements-deploy.txt` | 部署环境的 Python 依赖与版本 |
| `.gitignore` | Git 忽略大型数据、自动生成结果、临时文件等 |

`requirements` 在环境验证后锁定版本。驱动、GPU、CUDA、cuDNN 与 TensorRT 等信息另外写入环境报告，不只依靠 `pip freeze`。

当前 `requirements-train.txt` 已验证；`requirements-deploy.txt` 待 M0-05 确认兼容性后建立。上游源码由 `configs/source-lock.json` 与 `scripts/setup_source.py` 恢复，当前原始源码不重复上传到本项目 Git；后续修改以登记的 patch 保存。

Conda 环境不需要放在项目目录中。若后续使用 Git，重点版本管理代码、配置、文档和小型实验清单；大数据与全部训练权重另行管理。

## 三、configs：配置目录

为便于初学阶段查找，首版配置文件平铺，不再增加过多配置子目录。

| 文件 | 内容 |
| --- | --- |
| `configs/paths.yaml` | 项目路径规则与必要的外部路径 |
| `configs/desktop_public.yaml` | 公共子集训练数据配置 |
| `configs/desktop_mixed.yaml` | 公共 + 自采训练数据配置 |
| `configs/train_baseline.yaml` | 原模型训练超参数 |
| `configs/train_se.yaml` | 修改模型的训练参数；与基线保持可比 |
| `configs/yolo11n-seg-se.yaml` | 加入 P3-SE 的模型结构配置 |
| `configs/runtime.yaml` | 后端、模型位置、输入尺寸、阈值、摄像头、显示设置 |
| `configs/benchmark.yaml` | 固定测试输入、预热次数、重复次数、计时范围 |

路径管理约定：自己的脚本用 `Path(__file__)` 确定项目根目录，配置中的相对路径由统一路径函数解析。Ultralytics 数据 YAML 的 `path` 需要按其加载规则生成或转换成正确绝对路径。不能假设所有库都会按项目根目录自动解析相对路径。

本机路径示例可以写成 `E:/秋招/项目相关/YOLO`；Python 中若直接写 Windows 反斜杠路径，可使用 `Path(r"E:\秋招\项目相关\YOLO")`。

## 四、data：原始数据与整理后的数据

### 4.1 原始资料

| 子目录 | 放什么 |
| --- | --- |
| `data/raw/coco/annotations/` | 下载的原始 COCO 实例标注 |
| `data/raw/coco/images/` | 筛选并下载的原始 COCO 图片 |
| `data/raw/camera/videos/` | 自采原始摄像头短视频 |
| `data/raw/camera/frames/` | 抽取、筛选后的待标注图片 |
| `data/raw/camera/annotations/` | 标注工具导出的原始多边形标注 |

原始资料用于追溯与重新转换。训练程序读取下一节整理后的数据，避免临时调整和原始标注混在一起。

### 4.2 可以直接训练的数据

| 子目录 | 放什么 |
| --- | --- |
| `data/desktop/images/train/` | 训练图片 |
| `data/desktop/images/val/` | 验证图片 |
| `data/desktop/images/test/` | 测试图片 |
| `data/desktop/labels/train/` | 对应训练图片的 YOLO 多边形标签 |
| `data/desktop/labels/val/` | 对应验证图片的标签 |
| `data/desktop/labels/test/` | 对应测试图片的标签 |
| `data/desktop/splits/` | 不同训练方案与评估集合的图像清单 |
| `data/desktop/metadata/` | 类别映射、来源、拍摄组、划分与转换记录 |
| `data/fixed_checks/` | 固定的 30 张左右部署检查图片 |

首版用文件名前缀区分来源，例如 `coco_000123.jpg`、`camera_clip03_000015.jpg`。同名冲突在整理时处理，避免覆盖。

图片与标签必须对应：

| 图片 | 对应标签 |
| --- | --- |
| `data/desktop/images/train/camera_clip03_000015.jpg` | `data/desktop/labels/train/camera_clip03_000015.txt` |

每个实例一行多边形标签。真实背景图可以使用空标签；有目标却遗漏标注的图不能简单视为背景样本。

### 4.3 划分与元数据文件

| 文件 | 用途 |
| --- | --- |
| `splits/train_public.txt` | E1 使用的公共训练图片清单 |
| `splits/train_mixed.txt` | E2/E3 使用的公共 + 自采训练图片清单 |
| `splits/val_public.txt`、`val_camera.txt` | 分来源验证集合 |
| `splits/val_all.txt` | 统一验证与选择权重使用的集合 |
| `splits/test_public.txt`、`test_camera.txt` | 分来源最终测试集合 |
| `splits/test_all.txt` | 最终测试集合的组合清单 |
| `metadata/classes.json` | 0=杯子、1=瓶子、2=手机的唯一类别映射 |
| `metadata/manifest.csv` | 文件、来源、原图 ID、拍摄 group_id 与 split |
| `metadata/selection.json` | 公共子集选择规则、图像 ID 与种子 |
| `metadata/conversion_summary.json` | 标签转换、筛除与人工修正统计 |

这两套训练方案通过清单选择图片，不需要复制两份混合数据。训练、验证和测试图片应物理隔离，清单不得引入跨集合图片。

冻结清单建议保留项目根目录相对路径；使用前由数据脚本生成本机可读取的绝对路径清单。这样数据移动后能够重新生成，不必手工逐行修改。

`data/fixed_checks/` 用于开发和检查部署，不作为新的独立泛化测试证据。最终测试集不因反复检查后处理而进入训练池。

## 五、scripts：一次性任务与命令入口

脚本负责执行一个明确任务，可从项目根目录运行。

| 文件 | 职责 | 编写顺序 |
| --- | --- | --- |
| `scripts/check_env.py` | 检查版本、GPU、provider、摄像头 | 最先 |
| `scripts/prepare_coco_subset.py` | 筛选、下载、转换、类别重映射 | 数据阶段 |
| `scripts/extract_frames.py` | 抽帧并记录拍摄组 | 数据阶段 |
| `scripts/prepare_camera_labels.py` | 自采标注转换与数据整理 | 数据阶段 |
| `scripts/build_splits.py` | 按组划分、生成清单并防止重叠 | 数据阶段 |
| `scripts/check_dataset.py` | 图片、标签、类别和划分检查 | 数据阶段 |
| `scripts/visualize_labels.py` | 绘制标签轮廓、检查标注 | 数据阶段 |
| `scripts/train.py` | E1、E2、E3 的训练入口 | 训练阶段 |
| `scripts/evaluate.py` | 公共、自采和组合集合评估 | 训练阶段 |
| `scripts/check_modified_model.py` | 检查结构、梯度、加载和导出 | 模型修改阶段 |
| `scripts/export_onnx.py` | ONNX 导出与元数据记录 | 部署阶段 |
| `scripts/build_engine.py` | 根据 TensorRT 版本构建 engine | 部署阶段 |
| `scripts/compare_backends.py` | 输入、raw 输出、框与掩膜一致性检查 | 部署阶段 |
| `scripts/benchmark.py` | 固定输入的分段计时和结果导出 | 部署阶段 |

初期不用一次编写这些脚本，也不创建只有文件名却没有作用的大量空脚本。先写 `check_env.py`，之后根据实施路线增加。

## 六、deploy：可复用的推理代码

| 文件 | 职责 |
| --- | --- |
| `deploy/__init__.py` | 将目录作为 Python 包 |
| `deploy/paths.py` | 根据项目根目录统一解析路径 |
| `deploy/preprocess.py` | RGB 转换、letterbox、归一化与张量布局 |
| `deploy/postprocess.py` | 置信度筛选、NMS、框还原、掩膜还原 |
| `deploy/results.py` | 统一结果对象：框、类别、分数、掩膜、耗时 |
| `deploy/base_backend.py` | 定义各后端共有的加载与推理接口 |
| `deploy/torch_backend.py` | PyTorch raw 推理；作为性能与输出参考 |
| `deploy/onnx_backend.py` | ONNX Runtime 独立推理 |
| `deploy/trt_backend.py` | TensorRT engine 独立推理与内存管理 |

三个后端输出统一格式的原始候选和原型掩膜，使用共享前后处理。这样后端切换只改变模型执行部分，便于公平比较。

`scripts/` 是执行任务的入口，`deploy/` 是可被摄像头、评估和测速共同调用的实现。例如 `scripts/compare_backends.py` 调用三个 backend 比较结果，`app/webcam.py` 调用同样的 backend 实时显示。

## 七、app：摄像头应用

| 文件 | 职责 |
| --- | --- |
| `app/__init__.py` | Python 包入口 |
| `app/webcam.py` | 读取配置，串联采集、推理、绘制与退出 |
| `app/capture.py` | 摄像头与文件视频采集；必要时维护最新帧缓冲 |
| `app/render.py` | 绘制框、掩膜、类别、实例数、FPS 和耗时 |
| `app/recorder.py` | 截图、录像与输出日志管理 |

初期先用一个 `webcam.py` 跑通，代码变复杂后再拆出 capture、render、recorder。最终入口采用 `python -m app.webcam`，从项目根目录运行。

## 八、third_party：模型源码与结构修改

| 路径 | 内容 |
| --- | --- |
| `third_party/ultralytics/` | 锁定版本的官方源码，采用可编辑安装 |

SE 类与解析逻辑改在该源码中对应的模块文件。自定义连接关系保存在 `configs/yolo11n-seg-se.yaml`，修改说明与 patch 保存到 `docs/model_changes/`。

不要只修改环境中的临时 `site-packages` 文件，却没有记录对应源码。也不需要另外建立一个含义模糊的 `models/` 目录：结构配置放 configs、源码放 third_party、权重放 artifacts，职责明确。

## 九、artifacts 与 runs：模型文件和实验结果

### 9.1 artifacts 保存准备使用的模型

| 子目录 | 内容 |
| --- | --- |
| `artifacts/pretrained/` | 下载的 `yolo11n-seg.pt` 等初始权重 |
| `artifacts/E1/` | 公共数据基线选定权重与部署文件 |
| `artifacts/E2/` | 场景适配模型选定权重与部署文件 |
| `artifacts/E3/` | SE 修改模型选定权重与部署文件 |

每个完成部署的实验目录包含：

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 对应实验按验证规则选择的权重 |
| `best_fp32.onnx` | 固定尺寸 FP32 ONNX |
| `best_fp32.engine` | 目标电脑构建的 FP32 engine |
| `best_fp16.engine` | 精度验证合格时保存的 FP16/混合精度 engine |
| `model_meta.json` | 类别、输入输出、前后处理规范、源权重信息 |
| `build_info.json` | TensorRT、GPU、构建参数和实际精度模式 |

如 TensorRT 11 路线需要单独的强类型低精度 ONNX，则在同一实验目录增加对应 ONNX，并记录来源。文件名不代替真实 dtype/精度信息。

启动配置指向选定实验，例如 `artifacts/E2/best_fp32.onnx`。这样可以保留实验来源，避免多个 `best.pt` 反复覆盖。

### 9.2 runs 保留完整过程输出

| 子目录 | 内容 |
| --- | --- |
| `runs/train/` | 每次训练的完整结果目录 |
| `runs/eval/` | 每次评估的完整输出 |
| `runs/export/` | 导出与构建过程的中间结果 |
| `runs/benchmark/` | 原始计时记录、CSV/JSON 与逐次运行结果 |

一次训练用可追溯名称，例如 `runs/train/E2_seed42_20261002_143000/`。这是命名示例，不是按日期强制安排任务。

该训练目录通常保留 `weights/best.pt`、`weights/last.pt`、训练参数、指标表、曲线和验证可视化。训练结束后把选定文件整理到对应 artifacts 目录，原始 runs 继续保留。

完整实施方案原先用 `artifacts/best.pt` 等简写示例；本目录规划将其细化为 `artifacts/E2/best.pt` 等按实验组织的路径。后续命令与配置统一使用细化路径。

## 十、reports、logs、docs 与 demo

### 10.1 reports：分析与总结

| 子目录/文件 | 内容 |
| --- | --- |
| `reports/environment.md` | 硬件、依赖、安装与兼容性结果 |
| `reports/data/` | 标签可视化、类别统计、划分检查 |
| `reports/experiments/` | E1/E2/E3 对照表、错误案例与结论 |
| `reports/deployment/` | 输入输出一致性、框/mask 匹配、部署指标 |
| `reports/performance/` | 各后端均值、p50/p95、FPS 与分析 |
| `reports/stability/` | 长时间运行、资源占用与异常处理检查 |

### 10.2 logs：程序运行记录

| 子目录 | 内容 |
| --- | --- |
| `logs/environment/` | 环境检查原始日志 |
| `logs/build/` | ONNX 导出与 TensorRT parser/builder 日志 |
| `logs/app/` | 摄像头运行、耗时和异常日志 |

reports 是你归纳后的结论，logs 是程序直接记录的过程。训练工具已保存在 runs 中的日志不必重复复制一份。

### 10.3 docs：学习与复现材料

| 文件/子目录 | 内容 |
| --- | --- |
| `docs/项目实施方案.md` | 已确定的完整路线 |
| `docs/项目目录规划.md` | 本文件 |
| `docs/环境安装记录.md` | 安装命令、解决过的问题 |
| `docs/模型学习笔记.md` | 网络结构、输入输出、损失与掩膜机制 |
| `docs/部署学习笔记.md` | ONNX、TensorRT、预后处理与计时原理 |
| `docs/面试讲解.md` | 项目口述、实测结果与常见追问 |
| `docs/model_changes/` | 源码版本、修改说明、权重映射和 patch |

### 10.4 demo：演示输入与成果

| 子目录 | 内容 |
| --- | --- |
| `demo/input/` | 固定测速视频和演示用输入文件 |
| `demo/screenshots/` | 摄像头与三后端结果截图 |
| `demo/recordings/` | 保存的预测视频与面试演示视频 |

原始采集视频放 `data/raw/camera/videos/`，处理后的演示视频放 `demo/recordings/`。固定测速视频放 `demo/input/fixed_benchmark.mp4`，所有后端用同一输入。

## 十一、先建立什么，之后增加什么

第一步先建立全部一级目录，方便固定路径，但只开始编写以下内容：

1. `README.md`：记录项目目标和根目录。
2. `configs/paths.yaml`：约定路径解析。
3. `scripts/check_env.py`：确认显卡、软件与摄像头。
4. `docs/项目实施方案.md` 与 `docs/项目目录规划.md`：保存两份规划。
5. `third_party/ultralytics/`：安装并锁定源码。
6. `artifacts/pretrained/`：放初始模型。

之后按顺序填充：数据准备脚本与数据 → 训练和评估 → SE 修改 → ONNX 与 TensorRT → 摄像头应用 → 报告与演示。

## 十二、在 Windows 本地建立目录的命令

下面的 PowerShell 命令需要由你在自己的电脑执行。它只创建目录，不生成训练数据、模型权重或完整程序，不删除已有内容。现阶段只执行一级目录部分即可，子目录可在对应阶段建立。

```powershell
$projectRoot = 'E:\秋招\项目相关\YOLO'

$topLevelDirs = @(
    'configs', 'data', 'scripts', 'deploy', 'app', 'third_party',
    'artifacts', 'runs', 'reports', 'logs', 'docs', 'demo'
)

New-Item -ItemType Directory -Path $projectRoot -Force | Out-Null
foreach ($relativeDir in $topLevelDirs) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot $relativeDir) -Force | Out-Null
}
```

建立详细子目录：

```powershell
$projectRoot = 'E:\秋招\项目相关\YOLO'

$subDirs = @(
    'data/raw/coco/annotations', 'data/raw/coco/images',
    'data/raw/camera/videos', 'data/raw/camera/frames', 'data/raw/camera/annotations',
    'data/desktop/images/train', 'data/desktop/images/val', 'data/desktop/images/test',
    'data/desktop/labels/train', 'data/desktop/labels/val', 'data/desktop/labels/test',
    'data/desktop/splits', 'data/desktop/metadata', 'data/fixed_checks',
    'artifacts/pretrained', 'artifacts/E1', 'artifacts/E2', 'artifacts/E3',
    'runs/train', 'runs/eval', 'runs/export', 'runs/benchmark',
    'reports/data', 'reports/experiments', 'reports/deployment',
    'reports/performance', 'reports/stability',
    'logs/environment', 'logs/build', 'logs/app',
    'docs/model_changes', 'demo/input', 'demo/screenshots', 'demo/recordings'
)

foreach ($relativeDir in $subDirs) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot $relativeDir) -Force | Out-Null
}
```

不要提前在 `third_party/ultralytics/` 放空占位文件；后续 Git clone 会建立源码目录。规划中的 Python 脚本和配置文件应在实际编写阶段生成。

## 十三、后续运行路径示例

以下是待实现程序的目标接口示例，不表示脚本现在已经可以运行：

```powershell
Set-Location 'E:\秋招\项目相关\YOLO'
python scripts/check_env.py
python -m app.webcam --backend onnx --model artifacts/E2/best_fp32.onnx --source 0
python -m app.webcam --backend trt --model artifacts/E2/best_fp16.engine --source 0
```

后续正式实现均以本目录规划为路径依据。最终模型如果选择 E1 或 E3，则对应修改 runtime 配置和启动参数。
