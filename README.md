# YOLO 学习与项目入口

更新：2026-10-04。经用户确认，项目主线调整为两种固定盒装奶与一种瑞幸杯装咖啡的包装识别及实例分割，采用 YOLO11n-seg，自建数据、微调与独立部署。已冻结山姆全脂牛奶 200mL、伊利舒化 220mL、瑞幸咖啡杯三类；识别依据是可见包装，不判断液体成分。

**M0 6/6、M1 核心学习 6/6、M2 6/6、M5 6/6；M3 5/6、M7 1/6，主线 30/54 项，完整模块 4/9。** B1 采用 E1-A best / 20/5/5 / 实时 conf=0.25。**商品 ONNX 独立部署全模块已完成**：导出、共享前后处理、ORT CPU / CUDA 图片视频入口、30 输入配对与 val5 同条件质量对照均已验收。三个后端框 / 掩膜 mAP50-95=0.8205 / 0.7434，未测到部署降幅；104 项工程测试、261 历史文件保护通过。5 张四画面及指标图 / CSV / HTML 已生成。0.25 检出 7/8，瑞幸漏检、低门槛 CPU 裁剪边界残差保留；本批指标不代表广泛泛化。最终 test5 封存、摄像头仍用 PyTorch。下一项 M6-01 核对 TensorRT 构建条件；按演示需要可先接摄像头 ORT。见[M5-06 质量与可视化](docs/M5_ONNX独立部署与接口说明.md#m5-06-validation-quality)、[完整展示页面](demo/deployment/B1/M5-06/index.html)、[B1 档案](docs/M3_训练与实验规范.md#m3-06-baseline)。

历史夜间调试保留原 20 张的 14/6 划分与 3 轮短训练，conf=0.25 下调试权重六张均无检出。如今正式 E1 / E1-A 及摄像头接入已完成，采用 20 张训练 / 新 5 张验证，最终 5 张测试未运行；原 E0 命令保留，Live Camera 显式指定 `configs/live_products.yaml` 可加载 E1-A 三商品权重。摄像头结论为有限场景下基本可展示，仍有误检 / 瑞幸缺口，不作为正式准确率或长期稳定性。见[商品摄像头操作](docs/M0_环境与模型使用手册.md#live-products)及[历史调试结果](docs/M2_商品数据与标注规范.md#night-debug-results)。

**假期首版：**原 20 张作为正式训练池，新增 10 张用于验证 / 最终测试；图片和分割标签同步在线变换，不重复手画。新增采集及人工审核累计已用 10/30 张，剩余 20 张额度，不要求用满，也不扩充到 350 张。E1-A、摄像头、B1 冻结、商品 ONNX 导出 / raw 执行及共享预处理已完成，独立后处理也已验收，完整 ORT 图片 / 视频程序也已验收，30 个固定开发输入对齐与 val5 同条件质量对照均已通过，商品 ONNX 模块已收尾。若以后有明确演示缺口，再考虑少量针对性训练图 / E2，并另立版本。SE 和完整多后端验收按时间继续。见[小数据首版操作路线](plan/00_YOLO_分阶段推进与进度跟踪.md#small-data-route)。

**M1 核心学习已收尾：**用户正确回答 Precision / Recall、分割损失和独立评价问题，复述了当时预训练 / 后处理筛选与未迁移训练的边界。主聊天形成[项目口述、追问要点与实际问答记录](docs/模型学习笔记.md#m1-06)。如今已有正式三商品微调与验证证据，可结合实际漏检 / 轮廓问题练习项目解释，展示效果仍待改善。

## 常用入口

| 想查看什么 | 文档 |
| --- | --- |
| 当前进度与下一步 | [分阶段任务与进度跟踪](plan/00_YOLO_分阶段推进与进度跟踪.md) |
| 商品识别方向与假期首版 | [类别、数据、实验与日程](plan/00_YOLO_分阶段推进与进度跟踪.md#product-scope) |
| M2 标注与拍摄 | [类别、人工分工和试拍清单](docs/M2_商品数据与标注规范.md)、[采集 / 预标注 / 本地审核操作](docs/M2_商品数据与标注规范.md#m2-02-operation) |
| 新增 10 张审核结果与正式划分 | [M2-04：30 张原图、20/5/5 用途与下一步](docs/M2_商品数据与标注规范.md#m2-04-holdout) |
| 正式数据已可加载，下一步训练 | [M2-05、06：质量 / 增强 / 加载验收与命令](docs/M2_商品数据与标注规范.md#m2-05-06) |
| 正式训练规则、结果与下一步 | [M3：固定参数与命令](docs/M3_训练与实验规范.md)、[E1 原基线](docs/M3_训练与实验规范.md#m3-03-results)、[错误分析](docs/M3_训练与实验规范.md#m3-05-errors)、[E1-A：增强对照实测](docs/M3_训练与实验规范.md#e1a-augmentation) |
| 首版固定用什么、后续如何对照 | [M3-06：B1 基线档案](docs/M3_训练与实验规范.md#m3-06-baseline)、[机器清单](configs/baseline_products_v1.json)、[验收](reports/experiments/M3-06_products_v1_baseline.json) |
| 商品 ONNX 接口与部署进度 | [M5-01：实际导出、输入输出、CPU / CUDA 检查及命令](docs/M5_ONNX独立部署与接口说明.md)、[机器记录](reports/deployment/M5-01_B1_onnx_export.json) |
| 共享预处理代码、几何参数与复核 | [M5-02：输入转换与官方对照](docs/M5_ONNX独立部署与接口说明.md#m5-02-preprocess)、[机器记录](reports/deployment/M5-02_B1_preprocess.json) |
| 独立框 / 掩膜后处理与对齐 | [M5-03：NMS、掩膜还原、18 组对照](docs/M5_ONNX独立部署与接口说明.md#m5-03-postprocess)、[机器记录](reports/deployment/M5-03_B1_postprocess.json) |
| 独立文件推理与示例 | [M5-04：三后端、图片视频入口与运行命令](docs/M5_ONNX独立部署与接口说明.md#m5-04-file-inference)、[结果图片](demo/deployment/B1/M5-04_onnx_cpu.png)、[结果视频](demo/deployment/B1/M5-04_onnx_cpu.avi) |
| 固定输入的后端一致性检查 | [M5-05：30 输入配对与复核](docs/M5_ONNX独立部署与接口说明.md#m5-05-backend-parity)、[输入清单](configs/parity_products_B1.json)、[机器验收](reports/deployment/M5-05_B1_backend_parity.json) |
| 商品部署质量与可视化 | [M5-06：同条件指标、弱点与复核](docs/M5_ONNX独立部署与接口说明.md#m5-06-validation-quality)、[完整页面](demo/deployment/B1/M5-06/index.html)、[指标图](demo/deployment/B1/M5-06/metrics_comparison.png) |
| 夜间训练结果、白天从哪里接上 | [临时划分、实际加载、3 轮短训练与复查命令](docs/M2_商品数据与标注规范.md#night-debug-results) |
| 如何运行、检查和恢复 | [M0 环境与模型使用手册](docs/M0_环境与模型使用手册.md) |
| 实时摄像头双画面预览 | [Live Camera：启动、按键与类别筛选源码](docs/M0_环境与模型使用手册.md#live-camera) |
| E1-A 三商品摄像头 | [M7-01：新权重、启动方式与现场记录](docs/M0_环境与模型使用手册.md#live-products) |
| 已完成结果与验收证据 | [M0 阶段验收报告](reports/M0_阶段验收报告.md) |
| 模型原理学习 | [六阶段十八课学习手册](codex/00_YOLO学习路线与学习手册_终极版.md) |
| 当前 M1 怎么学、实测了什么 | [项目与面试入口](docs/模型学习笔记.md#interview-route)、[锁定模型实际观察报告](reports/model_notes.md) |
| 各版本改了什么 | [YOLO 各版本演进与主要改动速览](codex/03_YOLO各版本演进与主要改动速览.md) |
| 第一次从哪里学 | [第一课：从一张图理解目标检测](codex/01_第一课_从一张图理解目标检测.md)，跟读、手算，再完成五道练习 |
| 文件职责与最新清理状态 | [M5 收尾：171 缓存文件 / 29 空目录清理与部署复核](reports/maintenance/20261002_文件整理与清理清单.md#m5-cleanup) |

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

实时查看原图与分割结果，不录制；新商品模式显式指定配置：

```powershell
python app/live_camera.py --config configs/live_products.yaml
```

默认左侧原图、右侧模型结果，按 1/2/3 切换，Q / Esc 或关闭窗口退出；默认持续运行。商品配置加载 E1-A 三类网络，conf=0.25；可用 --conf 临时调整，保留剩余误检与瑞幸弱点。省略 --config 则保留 E0 杯子 / 瓶子 / 手机模式，E0 加 --all-classes 可显示 COCO 80 类。详见[商品模式与接入证据](docs/M0_环境与模型使用手册.md#live-products)。长期稳定性及多后端尚未验收。

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

此前 M0、M2 清理与复核记录继续保留。2026-10-04 M3 首版收尾已移除 199 个缓存 / 重复 / 已归档过程文件及 31 个空目录，9 个历史文件收进本地 ZIP；一处无文件的旧检查目录树因自动审批拒绝而保留。正式训练权重、数据、配置、实验 / 错误分析证据和 B1 清单不变，基线及正式 / 调试数据只读复核通过。详见[M3 历史清理记录](reports/maintenance/20261002_文件整理与清理清单.md#m3-cleanup)。清理完成时主线 24/54；接续 M5-01～06 后当前 30/54，M3 仍 5/6、E2 暂缓，最终测试封存。

M5 收尾已移除 171 个可重建 Python 缓存和 29 个缓存空目录，共 2,298,510 字节（约 2.19 MiB）；1,754 个保留文件 SHA / 修改时间一致，280 个部署相关文件哈希通过，M5 全链路只读复核通过。模型、数据、六份验收报告、参考数组、CUDA profile、图片 / 视频与 M5-06 展示页面均保留；没有新增历史 ZIP。详见[最新清理记录](reports/maintenance/20261002_文件整理与清理清单.md#m5-cleanup)。M5 仍为 6/6，下一步 M6-01。
