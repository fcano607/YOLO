# YOLO 参考资料目录与来源

整理日期：2026-09-30；课程入口更新：2026-10-01。学习顺序见 [终极版学习手册](../codex/00_YOLO学习路线与学习手册_终极版.md)。

你提供的 [YOLOv1 原文](YOLO.pdf) 保留不变。本次新增 10 篇 PDF，均从 arXiv 原文端点下载；另保存官方源码与文档快照。下载成功只表示资料可读，不表示每篇已经完整精读或模型已经运行。

## 1. 已下载论文

| 资料 | 本地 PDF | 年份 / 版本 | 页数 | 阅读定位 | 主要用途 |
| --- | --- | --- | --- | --- | --- |
| YOLOv1 | [原文](YOLO.pdf) | 2015 首投，v5（2016） | 10 | 基础卷，已有完整讲义 | 检测表示、损失、推理与评估 |
| [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) | [PDF](YOLO9000_2016_1612.08242.pdf) | 2016，v1 | 9 | 过渡选读 | anchor、直接位置预测与多尺度训练 |
| [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | [PDF](YOLOv3_2018_1804.02767.pdf) | 2018，v1 | 6 | 过渡重点 | 多尺度特征预测、逐候选类别与框 |
| [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) | [PDF](YOLOX_2021_2107.08430.pdf) | 2021，v2 | 7 | 现代基础重点 | 解耦头、anchor-free 与 SimOTA |
| [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) | [PDF](GFL_2020_2006.04388.pdf) | 2020，v1 | 14 | 损失专题 | 分布式回归与 DFL |
| [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755) | [PDF](TOOD_2021_2108.07755.pdf) | 2021，v3 | 12 | 分配专题 | 分类定位对齐与标签分配 |
| [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) | [PDF](YOLACT_2019_1904.02689.pdf) | 2019，v2 | 11 | 分割重点 | 共享原型与实例系数 |
| [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458) | [PDF](YOLOv10_2024_2405.14458.pdf) | 2024，v2 | 21 | 进阶重点 | 一致双分配与 NMS-free |
| [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748) | [PDF](YOLO26_2026_2606.03748.pdf) | 2026，v1 | 31 | 新版进阶主文献 | 检测、训练策略、实例分割与多任务 |
| [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) | [PDF](YOLOv7_2022_2207.02696.pdf) | 2022，v1 | 15 | 研究选读 | 重参数化、训练辅助设计 |
| [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524) | [PDF](YOLOv12_2025_2502.12524.pdf) | 2025，v1 | 13 | 研究选读 | 注意力结构与效率权衡 |

年份按 arXiv 首次投稿年记，不把它与正式会议出版年混写。YOLOX、YOLOv3 等技术报告与会议论文的出版状态需按各自原始记录区分。YOLO26 当前采用其 2026 年 arXiv 预印本，不声明已获某会议录用。

## 2. 下载与身份校验

所有新增文件已检查 `%PDF-` 文件头、可解析页数，以及前两页的 arXiv 编号；记录 SHA256，以便以后确认是不是同一份文件。未覆盖或重命名你提供的 `YOLO.pdf`。

完整记录见 [papers_manifest.json](papers_manifest.json)，包含原始地址、arXiv 版本、下载日期、页数、字节数和完整哈希。

| 文件 | 字节数 | SHA256 前 16 位 |
| --- | --- | --- |
| GFL_2020_2006.04388.pdf | 2882899 | `04a56cf0bd1c144f` |
| TOOD_2021_2108.07755.pdf | 3328021 | `ede5d01fe07bba7e` |
| YOLACT_2019_1904.02689.pdf | 8090124 | `223c83bd8f229bee` |
| YOLO26_2026_2606.03748.pdf | 9613058 | `8dec3ca2adbbc78c` |
| YOLO9000_2016_1612.08242.pdf | 5256083 | `6c5e99e00874eeeb` |
| YOLOX_2021_2107.08430.pdf | 871604 | `8b3d8689b6588788` |
| YOLOv10_2024_2405.14458.pdf | 1345990 | `db381148c9c2f5d8` |
| YOLOv12_2025_2502.12524.pdf | 2909966 | `c99d1ab39ed6dfc3` |
| YOLOv3_2018_1804.02767.pdf | 2456177 | `37049049b5e06f67` |
| YOLOv7_2022_2207.02696.pdf | 2269417 | `61827f8908649249` |

## 3. 官方源码和文档快照

仓库：`ultralytics/ultralytics`。读取日期：2026-09-30。固定 commit：`6956125297ae5a90f5d6d371cca93ea51d8320f9`。

保存到 `official_sources/`，只用于阅读和追溯；没有将这些文件作为项目依赖安装或执行。后续工程依赖仍须按实际环境锁定。

| 原始文件 | 本地快照 | 用途 |
| --- | --- | --- |
| [ultralytics/cfg/models/11/yolo11-seg.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/ultralytics/cfg/models/11/yolo11-seg.yaml) | [快照](official_sources/ultralytics__cfg__models__11__yolo11-seg.yaml) | P3/P4/P5 与 Segment 连接 |
| [ultralytics/nn/modules/head.py](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/ultralytics/nn/modules/head.py) | [快照](official_sources/ultralytics__nn__modules__head.py) | Detect/Segment 的字段与分支 |
| [ultralytics/utils/loss.py](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/ultralytics/utils/loss.py) | [快照](official_sources/ultralytics__utils__loss.py) | 框回归、DFL 和分割监督 |
| [ultralytics/utils/tal.py](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/ultralytics/utils/tal.py) | [快照](official_sources/ultralytics__utils__tal.py) | 任务对齐与参考点处理 |
| [docs/en/models/yolo11.md](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/docs/en/models/yolo11.md) | [快照](official_sources/docs__en__models__yolo11.md) | 当前工程模型官方说明 |
| [docs/en/models/yolo26.md](https://raw.githubusercontent.com/ultralytics/ultralytics/6956125297ae5a90f5d6d371cca93ea51d8320f9/docs/en/models/yolo26.md) | [快照](official_sources/docs__en__models__yolo26.md) | 新版模型的分支与推理说明 |

版本与文件校验记录见 [snapshot_manifest.json](official_sources/snapshot_manifest.json)。不要把 `main` 的未来内容与这份快照混为同一实现，也不要把源码注释中的示例形状当作本机实测。

## 4. 没有列入必修的资料

YOLOv7 和 YOLOv12 已下载作拓展，但不要求在现代基础课之前读完。其他 YOLO 版本、开放词汇检测和 Transformer 检测器，等遇到具体研究问题再补充。

YOLACT、GFL 和 TOOD 不是按 YOLO 版本命名的论文，纳入资料库是因为它们分别帮助理解原型分割、分布式框回归和任务对齐。不能把它们整个模型都说成 YOLO11 的实现。

## 5. 已有讲义与资料库的关系

- [YOLOv1 完整精读](../codex/YOLOv1_原论文精读与基础模型学习.md)：对应原始 YOLOv1。
- [现代基础过渡讲义](../codex/02_从YOLOv1到现代YOLO_检测与实例分割基础.md)：解释后续阅读的关键概念。
- [终极版学习手册](../codex/00_YOLO学习路线与学习手册_终极版.md)：统一阶段、十八课、练习与验收，区分已有材料和待编写专题。

提取文本与页面查看图位于 `tmp/pdfs/`，用于本轮核查。原始 PDF 和来源记录位于本目录；讲义引用以这里的原件为准。
