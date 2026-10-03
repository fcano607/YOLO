# YOLO 摄像头实时实例分割项目：目录规划

项目根目录：`E:\秋招\项目相关\YOLO`

2026-10-03 用户正式采用三种固定包装商品方向，现已根据试标成本确认小数据首版：现有 20 张起步，轻量在线增强和预训练微调；按需要少量补拍，新增人工审核累计最多 30 张（含留出，真实总量最多 50 张），不再按 350 张收集。M2-01 已冻结 classes.json / products.json，M2-02 的 20 张已验收；见[当前操作路线](00_YOLO_分阶段推进与进度跟踪.md#small-data-route)和[标注安排](../docs/M2_商品数据与标注规范.md#small-data-plan)。沿用现有 data/desktop/ 与代码路径，后续按任务实现配置，不增加空脚本。

本文件规划需要在你的 Windows 电脑中建立的目录与文件。2026-10-02 已完成 M0-02：必要一级目录、`configs/paths.yaml`、源码锁定清单、训练 requirements、路径解析与安装验收脚本已建立；后续数据、训练和部署脚本按任务逐步实现，表中其余文件仍是规划。实际进展见[进度跟踪](00_YOLO_分阶段推进与进度跟踪.md)。

同日已实现 `scripts/check_env.py`，生成 JSON 快照与 `reports/environment_check.md`；12:34 的复测已通过同一后端连续三次摄像头读取与正常释放，M0-03 已验收，见[检查使用说明](../docs/M0_环境与模型使用手册.md)。

同日 M0-04 已验收：新增 `scripts/prepare_e0.py`、`scripts/predict_e0.py` 与 `configs/e0.yaml`，预训练图片、文件视频和摄像头推理通过；产出保存在 `runs/E0/`、`demo/E0/`，见[E0 报告](../reports/M0_阶段验收报告.md)。

同日 M0-05 已验收：新增 `scripts/check_deployment.py`、`configs/deploy-precheck.yaml` 与 `requirements-deploy.txt`，完成 ONNX / ORT CPU/CUDA / TensorRT FP32 实际执行及同输入原始输出对照。模型和数组在 `artifacts/precheck/M0-05/`，小型执行证据在 `logs/deployment/`，见[部署报告](../reports/M0_阶段验收报告.md)；完整 M5/M6 仍待开展。

同日 M0-06 已验收，M0 模块完成：新增 `environment.yml`、`configs/constraints-runtime.txt`、`scripts/save_environment.py`、`scripts/verify_reproducibility.py`。归档位于 `logs/environment/M0-06_<时间戳>/`，新进程记录在 `logs/reproducibility/`，新模型 / 数组在 `artifacts/reproducibility/`；见[恢复说明](../docs/M0_环境与模型使用手册.md)和[验收报告](../reports/M0_阶段验收报告.md)。

同日新增 M7 实时预览原型 [app/live_camera.py](../app/live_camera.py)：同一摄像头显示原图和同帧 YOLO 分割结果，默认不录制；复用 `configs/e0.yaml` 与现有预训练权重。GPU 图片与模拟交互检查见 [logs/application/live_preview_check.json](../logs/application/live_preview_check.json)，真实摄像头窗口和持续运行仍待验证。操作与源码见[使用手册](../docs/M0_环境与模型使用手册.md#live-camera)，本次操作已登记[工作日志](00_YOLO_分阶段推进与进度跟踪.md#live-camera-record)。

同日 M1 已开始：新增 [scripts/inspect_model.py](../scripts/inspect_model.py)、[docs/模型学习笔记.md](../docs/模型学习笔记.md)、[reports/model_notes.md](../reports/model_notes.md)。本轮 JSON 和四张图集中在 `reports/model/M1/`，不另建逐层大数组目录。工程实操通过，个人学习验收仍待完成。

同日按用户安排先推进 M1-02：新增 [scripts/inspect_structure.py](../scripts/inspect_structure.py) 检查已加载模型的 YAML / 模块属性与连接，`reports/model/M1-02/` 仅保存结构记录和概览图；详细讲解合并到已有模型学习笔记第 2 节。M1-01 留待用户后续学习再讨论。

同日继续 M1-03：新增 [scripts/inspect_tensors.py](../scripts/inspect_tensors.py)，复用已有预处理与记录函数，执行两图关键层 / 分支 hook、普通 / export 对照和临时副本训练模式前向。`reports/model/M1-03/` 仅保存一份 [tensor_check.json](../reports/model/M1-03/tensor_check.json)，不保存完整大数组、额外视频或预测图；详细解释与复现命令合并到已有学习笔记第 3 节。工程检查与个人学习验收分别记录。

以下路径均相对于项目根目录，使用 `/` 表示子目录。例如 `configs/runtime.yaml` 在你的电脑上对应 `E:\秋招\项目相关\YOLO\configs\runtime.yaml`。

## 一、一级目录

| 目录 | 内容 | 创建时机 |
| --- | --- | --- |
| `configs/` | 数据集、训练、模型结构和推理参数配置 | 开始时 |
| `data/` | 原始资料、训练数据、标签、划分与固定推理样本 | 开始时 |
| `scripts/` | 环境检查、数据准备、训练、评估、导出与测速入口 | 开始时 |
| `deploy/` | 共享预处理、后处理、三种推理后端 | 开始时，可暂为空 |
| `app/` | 实时预览与 M2 本地商品轮廓审核；后续统一后端 / 录制模块 | 已有 live camera 和商品审核入口；M7 后续模块按任务实现 |
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
| `environment.yml` | M0-06 生成的 Windows Conda 基础恢复定义；GPU torch 与项目 pip 依赖分步安装 |
| `.gitignore` | Git 忽略大型数据、自动生成结果、临时文件等 |

`requirements` 在环境验证后锁定版本。驱动、GPU、CUDA、cuDNN 与 TensorRT 等信息另外写入环境报告，不只依靠 `pip freeze`。

当前 `requirements-train.txt` 与 `requirements-deploy.txt` 均已建立并在 M0 验证；日常操作与验收统一查阅合并后的 M0 手册和报告。上游源码由 `configs/source-lock.json` 与 `scripts/setup_source.py` 恢复，当前原始源码不重复上传到本项目 Git；后续修改以登记的 patch 保存。

Conda 环境不需要放在项目目录中。若后续使用 Git，重点版本管理代码、配置、文档和小型实验清单；大数据与全部训练权重另行管理。

## 三、configs：配置目录

为便于初学阶段查找，首版配置文件平铺，不再增加过多配置子目录。

| 文件 | 内容 |
| --- | --- |
| `configs/paths.yaml` | 项目路径规则与必要的外部路径 |
| `configs/products_base.yaml` | E1 首批审核商品数据配置，待实现 |
| [configs/augment_products.yaml](../configs/augment_products.yaml) | M2-03 已建立；轻量在线增强参数，已完成预览，实际训练未执行 |
| `configs/products_expanded.yaml` | E2/E3 定向扩充训练池，共用冻结验证 / 测试清单，待实现 |
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
| `data/raw/coco/annotations/` | 旧通用三类方案的预留路径；商品首版不下载 COCO 标注 |
| `data/raw/coco/images/` | 旧通用三类预留路径；不将 cup / bottle 图片直接改名成商品数据 |
| `data/raw/camera/videos/` | 自采原始摄像头短视频 |
| `data/raw/camera/frames/` | 抽取、筛选后的待标注图片 |
| `data/raw/camera/annotations/` | 模型草稿、逐图人工审核及少量批次预览；原图不涂画 |
| `data/raw/camera/sessions/` | 两批采集清单及合并审核列表，包含拍摄组、原图哈希、原提示与用户身份更正 |
| `data/raw/camera/annotations/drafts/` | M2-02 已生成首批模型轮廓 / 原 COCO 类别 / 分数草稿，不是训练真值 |
| `data/raw/camera/annotations/reviewed/` | 本地页面明确人工确认后保存审核 JSON / 分割 TXT；排除图不输出标签 |

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

商品首版使用实物 / 拍摄组 / 帧号命名，例如 `productA_clip03_000015.jpg`；保留原始文件，避免覆盖。E1/E2 由训练清单选择数据，复用审核照片，不复制多套数据目录。

图片与标签必须对应：

| 图片 | 对应标签 |
| --- | --- |
| `data/desktop/images/train/camera_clip03_000015.jpg` | `data/desktop/labels/train/camera_clip03_000015.txt` |

每个实例一行多边形标签。真实背景图可以使用空标签；有目标却遗漏标注的图不能简单视为背景样本。

### 4.3 划分与元数据文件

| 文件 | 用途 |
| --- | --- |
| `splits/train_base.txt` | E1 首批审核商品训练图片清单 |
| `splits/train_expanded.txt` | E2/E3 扩充商品训练清单，不引入验证 / 测试图片 |
| `splits/val_all.txt` | 冻结的商品验证拍摄组清单，用于调参与选择权重 |
| `splits/test_all.txt` | 冻结的独立测试拍摄组清单，仅用于最终评价 |
| `metadata/classes.json` | M2-01 已建立：三商品 ID / 英文名与实物引用，不沿用旧杯瓶手机映射 |
| `metadata/manifest.csv` | 文件、来源、原图 ID、拍摄 group_id 与 split |
| `metadata/selection.json` | 商品拍摄覆盖、抽帧 / 去重规则与划分种子 |
| `metadata/conversion_summary.json` | 草稿使用、人工修正 / 重画 / 补标数量、审核与转换统计 |
| `metadata/products.json` | M2-01 已建立：用户实物名称、已知容量 / 数量、冷杯与杯套，未知细节留空 |
| [metadata/dataset_inventory.json](../data/desktop/metadata/dataset_inventory.json) | M2-03 已建立：20 张来源 / 哈希 / 标签 / 拍摄组，尚未分配 split |
| [metadata/sampling_budget.json](../data/desktop/metadata/sampling_budget.json) | M2-03 已建立：基线 20 张、新增累计最多 30，当前用量 0；合并引用不重复计数 |

首批与扩充训练方案通过清单选择图片，不复制两份照片。训练、验证和测试图片应物理隔离，清单不得引入跨集合图片。

冻结清单建议保留项目根目录相对路径；使用前由数据脚本生成本机可读取的绝对路径清单。这样数据移动后能够重新生成，不必手工逐行修改。

`data/fixed_checks/` 用于开发和检查部署，不作为新的独立泛化测试证据。最终测试集不因反复检查后处理而进入训练池。

## 五、scripts：一次性任务与命令入口

脚本负责执行一个明确任务，可从项目根目录运行。

| 文件 | 职责 | 编写顺序 |
| --- | --- | --- |
| `scripts/check_env.py` | 检查版本、GPU、provider、摄像头 | 最先 |
| `scripts/check_deployment.py` | M0-05：固定 FP32 ONNX 导出、ORT CPU/CUDA 与 TensorRT 构建 / 执行、同输入原始输出对照 | 已实现，M0-05 |
| `scripts/save_environment.py` | 归档 Conda / pip / 硬件 / 源码 / 模型，生成恢复定义与当前版本约束 | 已实现，M0-06 |
| `scripts/verify_reproducibility.py` | 从项目外工作目录启动新进程，复现 E0 与 ONNX | 已实现，M0-06 |
| `scripts/prepare_e0.py` | 恢复并校验固定预训练权重及通用图片 | M0-04 已实现 |
| `scripts/predict_e0.py` | E0 图片、视频、摄像头推理，保存框、mask 与运行证据 | M0-04 已实现 |
| [scripts/annotate_product_drafts.py](../scripts/annotate_product_drafts.py) | 手动按空格采集原图 / 导入照片 / 原 COCO 全类别轮廓提议，保留来源；不生成审核真值 | M2-02 已实现，两批共 20 张采集与 GPU 草稿已运行；合并只引用图片，并保留原拍摄组 |
| `scripts/extract_frames.py` | 抽帧并记录拍摄组 | 数据阶段 |
| `scripts/prepare_camera_labels.py` | 转换人工审核商品标注，类别 + 多边形，不写置信度 | 数据阶段，待实现 |
| `scripts/build_splits.py` | 按组划分、生成清单并防止重叠 | 数据阶段 |
| [scripts/check_dataset.py](../scripts/check_dataset.py) | 原图 / 审核 / TXT 追溯、数据清单、锁定增强预览和累计预算 | M2-03 已实现；20 张及 60 次检查通过，正式划分与训练加载待后续 |
| `scripts/visualize_labels.py` | 绘制标签轮廓、检查标注 | 数据阶段 |
| `scripts/train.py` | E1、E2、E3 的训练入口 | 训练阶段 |
| `scripts/evaluate.py` | 冻结商品验证 / 测试组、各类别与条件评价 | 训练阶段 |
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

| 文件 | 职责 | 当前状态 |
| --- | --- | --- |
| [app/live_camera.py](../app/live_camera.py) | 使用 E0 预训练模型进行实时原图 / 分割结果预览，支持视图切换和三类 / 全类别选择，默认不录制 | 原型已实现；真实摄像头窗口待验证 |
| [app/annotate_products.py](../app/annotate_products.py) | 本地浏览器审核入口，明确确认后保存商品多边形标签与耗时统计 | M2-02 已实现；20 张人工标签已保存并验收 |
| [app/product_review.html](../app/product_review.html) | 多边形补画、拖点、改类、删非目标及逐图审核页面 | 已实现并完成浏览器操作检查 |
| [app/product_data.py](../app/product_data.py) | 类别读取、原图 I/O、多边形检查、标签导出和试标统计 | 已实现，12 项工程测试通过；包含闭合点与失败保存回归 |
| [app/product_dataset.py](../app/product_dataset.py) | 数据来源核验、按拍摄组统计、预算去重及原 Ultralytics 增强预览 | M2-03 已实现；新增 8 项工程测试，现有 12 项回归通过 |
| `app/__init__.py` | Python 包入口 | 规划 |
| `app/webcam.py` | 读取配置，串联采集、统一后端推理、绘制与退出 | 规划 |
| `app/capture.py` | 摄像头与文件视频采集；必要时维护最新帧缓冲 | 规划 |
| `app/render.py` | 绘制框、掩膜、类别、实例数、FPS 和耗时 | 规划 |
| `app/recorder.py` | 截图、录像与输出日志管理 | 规划 |

当前可运行入口是 `python app/live_camera.py`，从项目根目录运行；按键、参数与退出方式集中在[使用手册](../docs/M0_环境与模型使用手册.md#live-camera)。该入口目前使用 PyTorch / Ultralytics，尚未统一 ONNX Runtime / TensorRT 后端。后续按 M7 任务整理摄像头、文件视频、后端切换、截图与录制；`python -m app.webcam` 是后续规划入口，当前尚不可用。

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
| `artifacts/E1/` | 首批自采商品微调基线选定权重与部署文件 |
| `artifacts/E2/` | 定向扩充商品训练池模型及部署文件 |
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
| [logs/application/live_preview_check.json](../logs/application/live_preview_check.json) | 已有 Live Camera 原型检查结果，明确区分真实 GPU 图片与模拟摄像头 / GUI |
| `logs/app/` | 摄像头运行、耗时和异常日志 |

reports 是你归纳后的结论，logs 是程序直接记录的过程。训练工具已保存在 runs 中的日志不必重复复制一份。

### 10.3 docs：学习与复现材料

| 文件/子目录 | 内容 |
| --- | --- |
| `docs/项目实施方案.md` | 已确定的完整路线 |
| `docs/项目目录规划.md` | 本文件 |
| `docs/环境安装记录.md` | 安装命令、解决过的问题 |
| `docs/模型学习笔记.md` | 网络结构、输入输出、损失与掩膜机制 |
| `docs/M2_商品数据与标注规范.md` | M2-01 已建立：类别、轮廓规则、人工分工与试拍清单 |
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

按实际任务填充：当前 20 张整理 → 轻量在线增强 / 临时按组调试划分 → 加载与短训练 → 按需补 6–10 张独立留出并冻结版本 → E1 微调与验证 → 新商品摄像头和最终 ONNX。新增采集及人工审核累计最多 30 张；E2 定向补样本选做，SE、完整独立 ONNX / TensorRT 与稳定性按依赖推进。在线增强不重复保存多套图，合成实验只保存必要预览与来源；不创建空脚本凑目录。

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
