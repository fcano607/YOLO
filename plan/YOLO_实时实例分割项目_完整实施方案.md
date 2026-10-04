# YOLO 饮品包装识别与实时实例分割项目：完整实施方案

> 目标：国庆期间优先完成三种固定包装的自建小数据、轻量在线增强、迁移学习、留出检查、摄像头及 ONNX 首版。2026-10-03 用户确认现有 20 张起步、按需要少量补拍，新增人工审核累计最多 30 张；详细顺序见[小数据操作路线](00_YOLO_分阶段推进与进度跟踪.md#small-data-route)。完整技术路线继续保留，按实际进展验收。
>
> 按任务和验收标准推进，不按“必须学习七天”安排。每个阶段完成后进入下一阶段，训练等待时间可以与源码阅读、部署开发交叉进行。
>
> 编制日期：2026-09-30；2026-10-03 经用户确认调整商品方向。以下数据量、超参数、耗时与性能目标是建议起点，尚不是新商品实验结果。旧 E0 例子和记录保留，实际进展与假期安排查[任务清单](00_YOLO_分阶段推进与进度跟踪.md#product-scope)。

## 一、项目定义与最终成果

### 1.1 项目名称与类别

**基于 YOLO11n-seg 的饮品包装识别与实时实例分割系统。**

首版识别用户手头的两种固定盒装奶和一种瑞幸杯装咖啡，根据可见包装区分商品。M2-01 已按用户现有实物冻结三类与标注规则，见[数据规范](../docs/M2_商品数据与标注规范.md)。已知名称 / 容量已登记，详细厂家品牌、奶品版本和杯容量可补充，不阻塞首版。

| 已冻结 ID / 英文名 | 对象 | 当前登记 |
| --- | --- | --- |
| 0 / sam_whole_milk | 山姆全脂牛奶 200mL | 1 盒，厂家品牌可补充 |
| 1 / yili_shuhua | 伊利舒化 220mL | 1 盒，具体全脂 / 低脂版本可补充 |
| 2 / luckin_cup | 瑞幸咖啡杯 | 1 个冷杯，有杯套，识别品牌杯 |

类别 ID / 英文名已保存在 data/desktop/metadata/classes.json，实物登记在 products.json；训练 YAML 尚未生成。相同杯型不能仅凭外观区分美式与拿铁；奶液倒入无标识杯中也不属于首版识别范围。每类若只有一个实物，评价的是指定包装在新桌面场景中的表现，不能宣称覆盖整个品牌或不同包装批次。

当前 M0 / M1 使用原 COCO 80 类模型及 cup / bottle / cell phone 筛选，作为工程验证与学习例子保留。COCO 预训练来源用于新模型初始化，但没有三种具体商品标签；后续通过人工核实的自采标签真实训练，不能仅改名称表。

### 1.2 产品范围

完整路线支持图片、视频与摄像头，显示商品类别、分数、检测框、独立可见轮廓及本帧数量；记录模型 / 流水线耗时、实际 FPS 和帧龄，支持截图、录像与后端选择。首版先实现新商品 PyTorch 摄像头演示、留出评价与启动说明，时间允许时完成最终权重 ONNX 导出 / 执行；三后端完整后处理和质量 / 性能、SE 对照、持续稳定性按各模块单独验收。

“监测”限定为本帧识别与数量显示，跨帧去重累计、跟踪、行为或告警属于拓展。分割区域不直接提供体积、物理尺寸、液体成分或距离。

### 1.3 识别、检测与实例分割

整个画面输入同一 YOLO11n-seg，模型联合预测类别、框和掩膜信息，再经筛选、NMS 与轮廓还原得到每个实例。首版标签标注包装的整个可见外轮廓，商品身份根据实物与包装人工确认。

| 画面内容 | 新模型期望输出 |
| --- | --- |
| 奶品 A 第一盒 | 奶品 A 类别、分数、框及独立轮廓 |
| 奶品 A 第二盒 | 同类别，另一个框及独立轮廓 |
| 奶品 B 或指定瑞幸杯 | 各自商品类别、分数、框及独立轮廓 |
| 其他奶盒、普通杯或空桌面 | 非目标；作为干扰 / 背景检查误检 |

语义分割仅将相同类别的像素归为一类；实例分割还区分同类别的不同包装。类别识别、定位与轮廓预测不要求三个独立模型依次完成。实际效果以标注数据和留出评价确认，不能凭演示数量宣称准确率。

### 1.4 完成后应能解释

- 如何定义商品类别和可见识别范围，为什么保留实例分割。
- Backbone / 特征融合 / Segment、候选和原型怎样产生实例结果。
- 模型辅助标注怎样经过人工修边、改商品类别、补漏检成为训练标签。
- 如何按拍摄组划分，避免近重复泄漏；验证与最终测试各有什么用途。
- 预训练权重怎样迁移，商品微调、补样本和 SE 对照各改变什么。
- 如何验证加载映射、梯度、导出、后端一致性和质量 / 速度代价。
- 模型执行耗时、应用 FPS 与实际延迟的区别，未完成项怎样描述。

## 二、技术选型、范围与推进顺序

### 2.1 默认技术栈

| 部分 | 选择 | 选择理由 |
| --- | --- | --- |
| 开发语言 | Python | 便于利用已有 PyTorch 基础完成整个流程 |
| 主模型 | YOLO11n-seg | 保持此前项目选择，训练、修改和部署围绕一个基线展开 |
| 备用模型规模 | YOLO11s-seg | 仅在 n 模型精度不足、硬件与时间允许时比较 |
| 训练框架 | PyTorch + Ultralytics 源码安装 | 先调用成熟训练流程，再修改源码 |
| 权重来源 | 现有 COCO 预训练 YOLO11n-seg | 迁移基础视觉特征；不作为商品标签来源 |
| 商品数据 | 自采三种固定包装、模型辅助轮廓与人工审核 | 商品细分类别、可见分割及独立拍摄组评价 |
| 图像处理与显示 | OpenCV + NumPy | 实现采集、预处理、后处理和轻量演示界面 |
| 通用推理 | ONNX Runtime CPU / CUDA | 分离模型执行与训练框架 |
| GPU 优化 | TensorRT | 完成 engine 构建、独立推理和精度/性能对比 |
| 首个部署规格 | batch=1、固定 640×640 | 降低形状处理和多后端对齐难度 |
| 模型修改 | 后续 P3 分支 SE 公平对照 | 基线 / 数据错误分析后决定；不是假期首版前置条件 |

YOLO11 不是“必须追逐的最新模型”，这里选择它是为了保持项目主线稳定。后续升级模型版本属于另外一项实验。官方仍提供 YOLO11-seg 的训练、验证、预测和导出入口。[1]

### 2.2 任务顺序

| 阶段 | 工作 | 建议主动投入 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| A | 环境、预训练推理、部署兼容性预检查 | 2–4 小时 | GPU 可用，摄像头可读，原模型能导出 |
| B | YOLO 与分割机制、源码定位 | 3–5 小时 | 能说明输入输出并找到模型配置与分割头 |
| C | 现有 20 张整理、轻量增强、小批独立留出与冻结划分 | 按小数据工作实测更新 | 数据 / 标签正确且可加载；调试划分与独立留出用途明确 |
| D | E1 商品微调、评价；E2 定向扩充与错误分析 | 5–8 小时 | 基线权重、指标与错误清单可检查 |
| E | SE 修改、检查、对照训练 | 4–7 小时 | 修改模型可训练、可导出，有对照结果 |
| F | ONNX 独立推理与一致性检查 | 5–8 小时 | 不调用 Ultralytics 完成框与掩膜推理 |
| G | TensorRT 构建、独立推理与测速 | 4–7 小时 | engine 可运行，精度与速度有记录 |
| H | 摄像头整合、稳定性检查、成果整理 | 3–5 小时 | 三后端可演示，资料与复现说明齐全 |

表中其他工时为完整技术路线的早期参考；原基于 350 张估计的 32–56 / 34–60 小时、后续 29–51 小时和首版 15–25 小时已不作为当前小数据版本的工时承诺。C 阶段数据量与人工预算已缩减，剩余安排在首轮训练 / 补拍 / 部署实测后更新；训练和构建等待另计，不保证全部模块在假期完成。

推进原则：先在预训练模型上做一次 ONNX 与 TensorRT 的兼容性检查，尽早发现安装问题；再投入正式训练。正式模型出来后，将权重替换到已经跑通的部署流程中。

主线必须完成：基线训练、一次结构修改及对照、ONNX 独立推理、TensorRT 推理、摄像头演示、实验记录。INT8、C++、Web 服务、多摄像头和额外分类网络留到主线完成后。

## 三、阶段 A：环境与第一条可运行链路

### 3.1 先记录硬件，之后才锁定软件版本

已知有本地 NVIDIA 显卡和摄像头，但显卡型号、显存、驱动、现有 Python / CUDA 环境还未确认。不能据此保证某套安装命令适配。

先执行并保存：

```powershell
nvidia-smi
python --version
```

记录操作系统、GPU 型号、显存、驱动版本。`nvidia-smi` 中显示的 CUDA Version 表示驱动支持的 CUDA 上限，不等于已经安装的 Toolkit，也不等于 PyTorch 的运行时版本。

建立单独环境。建议先考虑 Python 3.11；如果所选 TensorRT wheel 不支持该版本，改用其支持的 Python 版本。

```powershell
conda create -n yolo-webcam python=3.11 -y
conda activate yolo-webcam
```

安装顺序：

1. 根据 PyTorch 官方安装选择器安装匹配驱动和 GPU 的 CUDA 版 PyTorch。[2]
2. 下载 Ultralytics 源码，先检出所选发布版本，记录 commit，然后 `pip install -e .`。
3. 安装 OpenCV、NumPy、PyYAML、ONNX 与数据转换依赖。
4. 首先让 ONNX Runtime CPU 推理成功。
5. 再安装与 CUDA、cuDNN 匹配的 ONNX Runtime GPU 包。
6. 按 TensorRT 支持矩阵选择版本与对应 Windows / Linux 安装包；将 Python bindings 与 `trtexec` 配齐。[3]

若使用 Windows，主线优先在原生 Windows 中运行摄像头和部署，减少额外的摄像头转发环节。遇到旧 GPU、新 GPU 或软件不兼容时，按照支持矩阵调整版本，完成同一项目主线。

ONNX Runtime 的 CPU 包与 GPU 包不要同时安装到同一环境，以免覆盖冲突。GPU 包还要核对 CUDA 与 cuDNN；cuDNN 8 与 9 不可混用。确认实际会话的 provider 和日志，避免把 CPU 回退当成 GPU 测速。[4]

以下是安装 Ultralytics 源码的顺序示例，正式训练前补上选定 tag 并记录 commit：

```powershell
git clone https://github.com/ultralytics/ultralytics.git third_party/ultralytics
cd third_party/ultralytics
# git checkout <选定并通过兼容性检查的发布 tag>
python -m pip install -e .
git rev-parse HEAD
cd ../..
```

`pip freeze` 只能记录 Python 依赖，另外还需保存驱动、GPU、CUDA 运行时、cuDNN、TensorRT 版本，以及修改后的源码 diff。

### 3.2 编写环境检查脚本

`scripts/check_env.py` 应输出：

- Python、PyTorch、Ultralytics、ONNX Runtime、TensorRT 版本。
- `torch.cuda.is_available()`、GPU 名称和总显存。
- `torch.version.cuda` 与 `torch.backends.cudnn.version()`。
- `onnxruntime.get_available_providers()`。
- 能否创建实际 CUDA 推理会话。
- 摄像头能否连续读出至少 30 帧。

检查结果保存为 `reports/environment.md`。所有安装通过后再冻结依赖，实验期间不无故升级。

### 3.3 第一次摄像头推理

先用预训练模型直接验证输入与输出：

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
for result in model.predict(
    source=0, stream=True, show=True, imgsz=640,
    conf=0.25, device=0, stream_buffer=False
):
    pass
```

摄像头索引通常从 0 开始，若未打开再检查权限、其他程序占用和 1 等索引。以上是库调用的入门验证，不是最终的独立部署实现。[5]

马上对该模型做一次固定输入 ONNX 导出。TensorRT 安装完成后对同一 ONNX 做构建检查。只需确认能构建与执行，不在此时完成全部后处理。

**阶段产出：**环境记录、依赖版本、摄像头截图、预训练 ONNX、TensorRT 兼容性检查记录。

**验收：**摄像头能显示分割结果；GPU 可用；ONNX 可加载；TensorRT 的可用版本与构建路径已经确定。

## 四、阶段 B：针对实施学习模型

### 4.1 学习范围

| 内容 | 需要掌握到什么程度 | 对应实践 |
| --- | --- | --- |
| 检测与分割 | 框和实例掩膜各表达什么 | 对同一张图查看 boxes 与 masks |
| 单阶段检测 | 多个位置并行预测，怎样筛选重复框 | 查看输出与 NMS 前后结果 |
| Backbone | 提取逐级特征 | 查看卷积与 C3k2 的尺寸变化 |
| Neck | 融合不同尺度特征 | 查看 Upsample、Concat 与 P3/P4/P5 |
| Head | 输出类别、框和掩膜相关信息 | 阅读 Detect、Segment、Proto |
| 分割机制 | 原型掩膜与实例系数如何组合 | 自己还原一张图的掩膜 |
| 训练目标 | 分类、框回归、DFL、分割损失的作用 | 对照训练日志与 loss 源码 |
| 评价指标 | Precision、Recall、box/mask mAP | 找到总指标与各类指标 |

不需要在正式训练前学完所有 YOLO 版本。先理解当前网络及输出接口，在遇到具体问题时阅读对应源码。

### 4.2 源码阅读路径

以锁定版本的实际文件为准，优先定位：

| 文件/目录 | 用途 |
| --- | --- |
| `ultralytics/cfg/models/11/yolo11-seg.yaml` | 主干、特征融合、分割头的连接关系 |
| `ultralytics/nn/modules/block.py` | C3k2、SPPF、C2PSA 等模块与新增模块 |
| `ultralytics/nn/modules/head.py` | Detect 与 Segment 的输出 |
| `ultralytics/nn/tasks.py` | YAML 解析与模型构建 |
| `ultralytics/models/yolo/segment/` | 分割训练、验证与预测 |
| `ultralytics/utils/loss.py` | 损失构成 |
| `ultralytics/utils/ops.py` | 框缩放、NMS、掩膜处理；部分版本 NMS 在独立文件 |
| `ultralytics/engine/exporter.py` | 导出设置与不同后端分支 |

官方 YOLO11-seg 配置使用 P3、P4、P5 三尺度特征，并通过 Segment 头进行实例分割。[6]

**阶段产出：**`reports/model_notes.md`，包含结构说明、关键张量尺寸和一次框/掩膜输出检查。

**验收：**能够解释“输入一帧图像，怎样得到多个目标的类别、框和各自掩膜”。

## 五、阶段 C：商品数据、辅助标注与冻结划分

### 5.1 首版数据规模（2026-10-03 小数据调整）

M2-02 已完成 20 张审核原图、30 个实例，含 16 张目标图 / 4 张无目标图。先将这批数据用于训练流程调试与学习能力检查，采用 COCO 预训练权重微调和轻量在线增强；按需要再采集少量独立留出或困难样本，不要求先扩充到 350 张。

| 用途 | 当前安排 |
| --- | --- |
| 现有原图 | 20 张；首轮按拍摄组建立临时调试划分 |
| 首批新增 | 建议 6–10 张，优先新场景验证 / 测试；执行时明确具体用途 |
| 人工预算上限 | 所有新增采集及人工审核累计最多 30 张，包含留出及后续训练图；真实总量最多 50 张，不要求用满 |
| 增强输入 | 训练时在线生成；图片与多边形同步变换，无需重复画边界 |
| 追加条件 | 只根据验证发现的缺口补训练图；E1 已能演示即可进入部署 |

旧 350 张、三类各 70 / 混合 90 / 非目标 50 以及固定 245 / 50 / 55 分配由本安排替代。真实原图、派生图、实例和独立拍摄组分别统计；增强变体不能算独立场景。实际顺序及谁负责见[操作路线](00_YOLO_分阶段推进与进度跟踪.md#small-data-route)。

### 5.2 商品登记与自采

M2-01 已冻结类别及轮廓、遮挡和非目标规则；详细商品资料可在试标时补充。杯身、装上的杯盖和贴合的杯套为同一个包装实例；严重遮挡、无法区分类别或复杂断开轮廓的图片首批单独放置。首版覆盖现有实物，瑞幸按品牌杯识别，不判断内部饮品种类。

拍摄覆盖多背景、光照、距离、角度、单个 / 混合目标与部分遮挡。各类别使用相近的条件，避免奶品 A 总在一个背景、奶品 B 总在另一个背景。非目标图片加入其他奶盒和普通杯，检查是否仅靠外形识别。

单商品分批拍摄并记录实物 ID，可预填商品类别；混合图需逐实例核实。原 COCO cup / bottle / cell phone 不包含新商品身份，[7] 仅保留为预训练来源资料，不再建立旧公共三类训练子集。

### 5.3 模型辅助标注与人工审核

原始照片 → 现有 YOLO11n-seg 提议框 / 掩膜 → 人工逐实例确认商品名、修正可见轮廓、补漏检并删除误检 → 标准标签 → 加载 / 叠加检查。模型可能漏检奶盒，不能预设所有包装都已被检出；未检出目标仍需手工标注。

模型只提供草稿，商品类别由人工提供。可以在审核后复用正确轮廓并将该实例改成对应商品 ID；不得把整个 COCO cup / bottle 类全局改名。预测置信度不写入标准分割标签，改成 1 不等于真值审核。保存未涂画的原照片，预测效果图只作检查，不输入训练。

草稿与审核标签分开；每图全部目标需标注。验证 / 测试标签也要人工核实，未经修正的自动标签不能充当独立评价标准。辅助标注节省时间的结论来自试标计时，不预先宣称节省比例。可参考官方自动标注资料，[17] 实际先复用当前模型与依赖，不为试标默认新增模型环境。

### 5.4 标签、分组与元数据

每个实例一行：[8]

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

坐标归一化到 0–1，至少三个顶点；矩形检测框不能替代真实分割轮廓。本地 Results.save_txt 默认 save_conf=False，可作为轮廓草稿导出的参考，实际实施需处理类别重映射、重复写入与人工复核。

可见区域沿物体真实轮廓标注；杯套、孔洞与复杂多连通轮廓在 M2-01 明确。目标漏标不能当作背景，非目标 / 空背景图才使用空标签。保留原始审核状态、类别映射版本、实物 ID 与拍摄 group_id。

记录实物 ID，先按拍摄组建立临时调试 train/val，当前两批互相关联，不当独立效果证据。正式首版优先新增两个独立场景 / 时段的留出组，一组验证、一组最终测试；不随机拆近重复图凑比例，独立组不足时只报告验证或定性演示。训练 / 验证 / 测试先划分，再做增强；增强 / 合成源图只选训练池，派生图继承来源用途。E1/E2/E3 共用固定留出，E2 只增加训练图，测试不用于调参或补样本。单实物条件下只评价指定包装。

### 5.5 数据配置与验收

已实现 `configs/products_base.yaml` 选择正式 20 张训练池与各 5 张验证 / 测试清单；`configs/products_expanded.yaml` 后续选择扩充池，共用留出，继续沿用 `data/desktop/`，不复制两套照片。

以下为 2026-10-04 已建立并通过实际加载验收的正式 YAML，使用 products-v1 三类映射；省略 path，让加载器按 YAML 所在目录定位清单。临时调试版本单独保留，正式 test 不指向 train：

```yaml
train: ../data/desktop/splits/train_base.txt
val: ../data/desktop/splits/val_all.txt
test: ../data/desktop/splits/test_all.txt
names:
  0: sam_whole_milk
  1: yili_shuhua
  2: luckin_cup
```

检查图片可读、标签配对、类别唯一、坐标 / 顶点有效、全部目标已标注、近重复无跨组泄漏。小数据首版逐张检查全部真实原图（最多 50 张），取消“至少 50 张”要求；抽查代表性增强图的掩膜同步及品牌可辨性，统计每类实例、场景条件、来源与人工成本。

**阶段产出：**商品登记、审核标签、数据配置、冻结清单、拍摄组元数据、叠加图及数据报告。尚未实施的脚本与配置仅按任务创建，不先堆积空文件。

**验收：**加载器可读；商品类别和可见轮廓正确；组间独立；人工审核与数据版本可追溯。

### 5.6 小数据增强策略

优先使用锁定源码现有的在线训练增强，原照片只需人工标注一次；颜色变换不改变标签，几何变换对图像及多边形同步计算。可核对 [v8_transforms / RandomPerspective](../third_party/ultralytics/ultralytics/data/augment.py) 和[官方增强说明](https://docs.ultralytics.com/guides/yolo-data-augmentation/)。

M2-03 已将轻量策略保存为 [configs/augment_products.yaml](../configs/augment_products.yaml)，完成 20 张 / 60 次锁定源码诊断，见[实测记录](../reports/data/M2-03_dataset_check.json)。2026-10-04 夜间已进一步建立 debug-v1 的 14/6 按组配置，全部实际训练加载与 3 轮 GPU 短训练通过；三商品头、12 次更新及 best/last 重载已有[证据](../reports/experiments/M3-01_products_debug_v1.json)。conf=0.25 下六张调试验证图均无检出，正式 E1 与独立留出仍待执行；预览和关联场景调试不属于正式商品效果评价。首轮关闭镜像和强色相，Mosaic / Copy-Paste / OpenCV 换背景仍选做，最新接续步骤见[夜间结果](../docs/M2_商品数据与标注规范.md#night-debug-results)。

2026-10-04 白天接续：M2 6/6 已完成，正式 [products-v1.json](../data/desktop/splits/products-v1.json) 冻结 20/5/5、30 原图 / 44 实例，正式质量 / 实际加载及 60 次训练增强通过；90/90 次目标保留，空标签批次 12 次仍为空，[正式 YAML](../configs/products_base.yaml) 与加载契约齐备，34 项工程测试通过。新增额度仍为 10/30、剩余 20，当前无需继续拍照，接下来固定 E1。正式微调与最终模型测试未执行，最终 test 只做标签 / 路径验收，不用于调参、选权重或追加训练数据决策；每类仍同一个实物，只评价固定包装在少量新场景的表现。最新结果见[数据验收](../docs/M2_商品数据与标注规范.md#m2-05-06)，上段保留夜间历史证据。

不为凑数量落盘生成几百个文件。验证 / 测试仅做必要的尺寸与输入预处理，禁止混入训练原图的增强版本；最终测试在模型选择后再使用。用户新增人工预算累计最多 30 张，启用增强本身不要求重复手画。

## 六、阶段 D：基线训练、评估和错误分析

### 6.1 先做小规模检查

先使用当前 20 张建立临时按组调试清单，实际训练池大小以清单为准，不要求另补到 20–50 张训练图。训练 1–3 epoch，检查加载、有限损失、显存、保存 / 重载与掩膜输出；流程通过后，再按验证条件和预算完成 E1 微调。

必要时用极少量图片做过拟合检查，验证流程具备学习能力；这只用于排错，不能用于评价泛化。

### 6.2 正式实验设置

2026-10-04 M3-02 固定首轮 [train_baseline.yaml](../configs/train_baseline.yaml)，接续 M3-03 已执行 50 轮 / 150 次更新，best/last 保存重载与固定 val 初步评价通过。当前参数如下；后续变更需记录新配置并重新预检查：

| 参数 | 起始设置 | 调整原则 |
| --- | --- | --- |
| 初始化 | `yolo11n-seg.pt` | 使用预训练权重微调 |
| 输入尺寸 | 640 | 正式实验先固定 |
| epochs | 50 | M3-03 已实测完成，无早停 |
| batch | 8 | GPU 实际三批 8/8/4 前向 / 反向通过 |
| optimizer | AdamW | 明确指定，便于对照 |
| lr0 | 0.001 | 微调起点；不盲目叠加多项调整 |
| weight_decay | 0.0005 | 对照实验保持相同 |
| patience | 0 | 关闭基于指标的早停，首轮完整 50 轮 |
| seed | 42 | 固定划分与主实验种子 |
| workers | 0 | Windows 首轮实际加载已通过 |
| AMP | 关闭，FP32 | 与已验证调试路径一致，损失 / 梯度有限 |
| nbs / warmup | 8 / 3 轮 | accumulate=1，warmup=9 次迭代，实际更新 150 次 |
| 学习率调度 | cos_lr=true、lrf=0.01 | 明确保存调度与逐轮学习率 |
| 增强 | 已验收轻量 HSV / 缩放 / 平移 / 小角度旋转，镜像与拼图关闭 | 沿用 augment_products.yaml，不改人工标签 |
| close_mosaic | 0 | 首轮 mosaic=0 |

以上是首版执行配置，是否能展示需正式训练后验证。锁定源码默认分割 fitness 为 box 与 mask mAP50-95 之和；本项目用 [MaskFirstSegmentationTrainer](../app/product_trainer.py) 按 val 的 mask mAP50-95 选 best，并列取较晚一轮，第三方源码不变。评价 conf=0.001 / NMS IoU=0.7 / max_det=300，展示 conf=0.25；最终 test 不进入训练选权重或 M3 评价入口。

已实现 Windows 主入口。以下预检查、正式训练与独立验证均已执行：

```powershell
python scripts/train.py --config configs/train_baseline.yaml --preflight
python scripts/evaluate.py --check
```

```powershell
# 本次正式训练已完成；重跑需另给新的 --name
python scripts/train.py --config configs/train_baseline.yaml
# 训练完成后检查固定验证池
python scripts/evaluate.py --run E1_products_v1_seed42
```

GPU 预检查实际 20 张 / 3 批有限损失及反向梯度、val 5 张双入口执行通过；allocated 约 2.19 GiB、优化器更新 0 次、没有保存新权重。新增 7 项实验规则测试及原 34 项均通过，配置 / 数据 / 实现和环境核对齐备，见[M3 规范](../docs/M3_训练与实验规范.md)与[机器记录](../reports/experiments/M3-02_E1_setup.json)。默认 `scripts/train.py` 仍用历史 debug 配置；正式 E1 必须显式指定配置，E2/E3 的配置和入口适配仍待实现。

接续 M3-03：E1 完成 50 轮 / 150 更新、调用 45.66 秒，val box / mask mAP50-95=0.4735 / 0.4658；M3-05 发现奶盒低分和轮廓越界。E1-A 在同数据 / 预算下强化角度与缩放，完成 50 轮 / 150 更新、调用 45.15 秒，val box / mask mAP50-95=0.8130 / 0.7807，0.25 匹配 6/8。M7-01 现场反馈 0.25 误检减少、三类基本能检出，仍有误检 / 瑞幸弱点；M3-06 已冻结 B1。接续 M5-01 已导出静态 640 / batch1 / FP32 商品 ONNX，图检查、真实三类接口和已有 val 三图的 ORT CPU / CUDA raw 对照通过，profile 确认实际 CUDA 节点执行；65 项测试、219 文件保护通过，无新照片 / 训练 / test 推理。当前 M3 5/6、M5 4/6、M7 1/6、主线 28/54，E2 暂缓。M5-02 共享预处理已实现，官方 5 张 val / 12 个尺寸输入及 3 份保存输入完全一致，70 项测试通过；M5-03 独立后处理 18 组对照通过，候选 / 框和二值掩膜一致，80 项测试通过；M5-04 完整 ORT / 参考后端与图片视频入口也已验收，三图原图对齐、GPU / 文件读写释放通过，88 项测试通过；压缩视频一帧伊利掉门槛漏检保留。下一步 M5-05 固定开发输入配对，部署 mAP 尚待后续，最终测试继续封存。详见[M5 实际接口与验收](../docs/M5_ONNX独立部署与接口说明.md)、[B1 档案](../docs/M3_训练与实验规范.md#m3-06-baseline)与[摄像头记录](../docs/M0_环境与模型使用手册.md#live-products)。

### 6.3 必须记录什么

- 数据版本、配置、种子、源码 commit、实际训练时长和显存。
- 总体与各类 box mAP50、box mAP50-95、mask mAP50、mask mAP50-95。
- Precision、Recall、损失和验证曲线。
- 最优权重的选择依据，固定采用同一种验证集规则。
- 典型错误图片：漏检、误检、类别混淆、框偏移、掩膜不完整或越界。

mask mAP50-95 为核心精度指标，检测指标用于解释定位与分类问题。摄像头 FPS 属于应用性能，不能代替精度评估。

验证集用于调参和选择权重。测试集在训练策略和模型选择完成后统一评估，报告各商品及常规 / 遮挡 / 光照等条件；同一实物跨场景与新包装泛化必须区分。

### 6.4 错误对应措施

| 观察到的错误 | 优先检查或调整 | 对照方式 |
| --- | --- | --- |
| 商品在新桌面场景表现较弱 | 背景、光照、角度覆盖及标签质量 | E1 首批训练 vs E2 定向扩充，验证 / 测试组固定 |
| 小物体漏检 | 目标像素面积、尺度覆盖、置信度 | 在验证集先查召回，必要时补小目标数据 |
| 暗光或背光漏检 | 训练样本与增强范围 | 增加对应训练样本，保持验证集固定 |
| 背景误检 | 负样本与标签遗漏 | 加入真实背景图，观察误检与召回变化 |
| 掩膜不完整 | 标签、遮挡、后处理和分辨率 | 先区分训练误差与部署还原错误 |
| 指标提升但摄像头卡顿 | 推理、传输或后处理瓶颈 | 分段计时后再决定优化点 |

先做有依据的数据和训练调整。结构修改使用第五阶段定义的单一方案，避免同时改数据、损失、模型和输入尺寸。

## 七、阶段 E：完成一种结构修改和对照实验

### 7.1 修改目的与边界

结构实验采用 **P3 分割输入分支增加一个 SE 通道注意力模块**，在商品基线和数据错误分析后开展，不作为假期首版前置条件。SE 根据特征的全局统计生成通道权重，重新加权特征。[11]

项目中的待验证假设是：在复杂桌面背景中，对较高分辨率 P3 特征进行通道重加权，可能改善部分目标的识别与掩膜质量。是否有效必须用实验确认。YOLO11 已含 C2PSA，因此不能仅凭“增加注意力”推断一定提升，也不能将已有 SE 方法宣称为原创算法。

只增加一个 SE，保持类别、输入尺寸、训练数据、主要训练超参数与评估规则不变。暂不同时加入 CBAM、P2 分支、新损失等。

### 7.2 实现路线

SE 使用 `AdaptiveAvgPool2d(1) → 1×1 Conv → ReLU → 1×1 Conv → Sigmoid → 逐元素相乘`，输出形状与输入一致。缩减比设为 16，内部通道下限为 1。

```python
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        hidden = max(1, c1 // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)
```

实施顺序：

1. 在锁定的 Ultralytics 源码中加入 SE 类，并通过 `nn/modules/__init__.py` 导出。
2. 在 `nn/tasks.py` 导入 SE，使 YAML 模块解析能够找到它。
3. 在 `parse_model` 增加专用分支：从来源层取得 `c1=ch[f]`，向 SE 传入输入通道，令 `c2=c1`。不要把 SE 当成会改变输出通道的普通 Conv 处理。
4. 复制原模型 YAML，明确使用 n 规模。
5. 在原 Segment 之前追加 SE，而不是插入主干中间，以减少层索引变化。
6. 将 Segment 的 P3 来源改为 SE 输出，保留 P4、P5 来源。

以官方配置原始 P3/P4/P5 索引 16、19、22 为例，末尾两层可写为：[6]

```yaml
# 前面的原始 0–22 层保持相同
- [16, 1, SE, [16]]                  # 新增第 23 层；args 中的 16 是 reduction
- [[23, 19, 22], 1, Segment, [nc, 32, 256]]  # 原 Segment 移至第 24 层
```

正式实施必须核对选定版本的实际索引。YAML 加载时明确设置 n 的尺度，避免因自定义文件名造成错误规模推断。[12]

### 7.3 预训练权重加载是必查项

追加 SE 后，原 Segment 从 23 移到 24。默认按名称加载权重可能会导致分割头无法对应，影响实验公平性。

处理方式：

- 原 0–22 层直接沿用匹配权重。
- 将原权重中 `model.23.*` 的分割头键映射到新 `model.24.*`。
- 仅复制形状一致的张量。
- 从 80 类变成 3 类时，类别输出层等形状变化部分需要初始化，基线和改进模型都采用同样规则。
- SE 使用固定种子的初始化。
- 保存已加载、缺失、形状不匹配的权重键与数量，确认共享层和掩膜相关层没有无意丢失。

不能仅凭“程序没报错”就认为预训练权重正确加载。

### 7.4 正式训练之前的检查

1. 构建模型，打印规模、参数量和连接来源。
2. 用 1×3×640×640 输入完成前向，检查框与原型掩膜输出。
3. 用真实小批量数据完成一次反向与参数更新，确认 SE 的梯度有效。
4. 检查损失不是 NaN/Inf。
5. 导出 FP32 ONNX，并通过 `onnx.checker` 和 ONNX Runtime 单图推理。
6. 用该 ONNX 尝试 TensorRT 解析/构建。
7. 全部通过后才投入正式对照训练。

### 7.5 实验矩阵

| 实验 | 模型 | 训练数据 | 用途 |
| --- | --- | --- | --- |
| E0 | 原 COCO 预训练 YOLO11n-seg | 无商品微调 | 保留旧工程演示与输出验证；不能当成新商品精度基线 |
| E1 | 三商品 YOLO11n-seg | 首批自采审核训练池 | 商品微调基线，假期优先 |
| E2 | 三商品 YOLO11n-seg | 首批 + 针对验证错误新增的训练样本 | 衡量定向数据扩充整体效果 |
| E3 | 三商品 YOLO11n-seg + P3-SE | 与 E2 相同 | 同条件结构对照，首版后按时间开展 |

E1/E2 都从同一预训练来源开始，差异表示数据扩充方案整体效果，数据量和优化步数同时记录；验证 / 测试组保持不变。E2/E3 使用同数据、同初始化规则和训练预算，才用于结构对照。

E2/E3 都从同一个 COCO 预训练来源开始，采用相同的三类输出初始化规则；不要只让 E3 从已经完成场景微调的 E2 权重继续训练。两者使用同样划分、种子、输入尺寸、训练预算、优化器、增强和权重选择规则。尽量固定 batch；若显存迫使改变，应重跑可比较配置或明确说明限制。首轮同一 seed 可用于学习项目；如果提升很小且要宣称稳定提升，再补 2–3 个 seed。

保存所有对照结果。最终部署在验证集上满足精度与速度要求的模型；即使 E3 没有提升，仍保留修改代码与负结果分析。测试集不参与最终模型选择。

**阶段产出：**自定义模块、模型 YAML、权重加载记录、训练/导出检查记录、E1/E2/E3 对照表。

**验收：**你能够解释改了哪里、为什么改、如何控制变量、有没有改善以及代价是什么。

## 八、阶段 F：ONNX 独立部署

### 8.1 两次导出

第一次导出预训练模型做兼容性检查；第二次导出最终选中的项目权重。正式链路使用：固定 640×640、batch=1、FP32、无内置 NMS 的原始输出。

```python
from ultralytics import YOLO

model = YOLO("artifacts/best.pt")
model.export(
    format="onnx", imgsz=640, batch=1,
    dynamic=False, nms=False, simplify=False, opset=17
)
```

`opset=17` 是首轮兼容性尝试值，需由所选导出器、ONNX Runtime 和 TensorRT 实际验证。先关闭图简化，检查通过后可另行比较简化图；不一次叠加动态尺寸、FP16 与内置 NMS。[13]

导出 API 随 Ultralytics 版本改变。执行前核对所锁定版本的参数含义，特别是精度参数、NMS 与输出布局。ONNX 检查通过不等于真实推理与后处理正确。

同时保存 `model_meta.json`：类别名称、nc、nm、输入尺寸、颜色顺序、归一化、letterbox 参数、输出名/shape、NMS 阈值、掩膜规则、版本和权重摘要。

### 8.2 独立的含义

最终 `deploy/onnx_backend.py` 使用 ONNX Runtime 执行图，使用 NumPy/OpenCV 完成预处理与后处理，不调用 `YOLO(...).predict()`，不依赖 PyTorch 来执行模型或还原掩膜。

允许在训练/导出环境使用 PyTorch；若 GPU DLL 需要预加载，优先采用所选 ORT 版本官方支持的预加载或运行时依赖安装方法。部署环境依赖单独记录。

### 8.3 预处理实施步骤

1. 从 OpenCV 获取 BGR 原图，记录 H、W。
2. 将 BGR 转为 RGB。
3. 计算等比例缩放比例 `r=min(640/W, 640/H)`。
4. 按相同舍入规则缩放，补边到固定 640×640；保存真实 left/top/right/bottom 值。
5. 补边值使用与参考实现一致的值，例如 114。
6. 转为 float32 并除以 255。
7. HWC 转 CHW，增加 batch 维度，形成连续内存的 1×3×640×640 张量。

固定输入时不能启用会将实际尺寸缩成其他矩形大小的自动补边设置。对齐时参考 PyTorch 路径也必须使用同一张预处理张量，不能各自默认预处理后直接比较。

### 8.4 读取真实输出

对首版 YOLO11-seg、三类、nm=32、640 输入、原始输出，预期候选输出常见为：

- `pred`：1×(4+3+32)×8400，即 1×39×8400。
- `proto`：1×32×160×160。

三层 80×80、40×40、20×20 的位置总数为 8400。这里的形状是当前结构与设置下的预期；运行时必须读取模型输出信息，不能硬编码假定所有版本都相同。

候选中的内容是框、类别分数和掩膜系数。此 YOLO11 原始输出没有另一个需要相乘的 YOLOv5 式 objectness 字段；框也需要确认是解码后的 cx、cy、w、h。用实际输出和锁定版本 Head 源码核验。[6][10]

### 8.5 框的后处理

1. 将候选维度整理为 N×(4+nc+nm)。
2. 每个候选取最大类别分数及其类别；首版采用单标签规则。
3. 根据 conf 阈值过滤，保留对应 32 维掩膜系数。
4. cxcywh 转为 xyxy。
5. 使用逐类别 NMS，避免不同类互相抑制。
6. 首版 conf=0.25、NMS IoU=0.7、max_det=100；参考路径和所有部署后端保持一致。
7. 保留原候选索引，使框、类别、分数和掩膜系数一一对应。
8. 用补边与缩放参数还原原图框并裁剪到原图范围。

这组阈值只是起点。通过验证集选择后固定，不能在部署对比中为每个后端分别选阈值来掩盖差异。

### 8.6 掩膜的后处理

保留 K 个实例后的系数矩阵 A 为 K×32，原型 P 为 32×160×160。核心组合为：

$$
L=A\,\mathrm{reshape}(P,32,160\times160)
$$

将 L 还原为 K 张掩膜 logits。Sigmoid 后以 0.5 判定等价于 logits 以 0 判定，但插值、裁剪、阈值操作的先后顺序会改变边缘。

首版明确对齐锁定版本的 `retina_masks=True` / `process_mask_native` 路径：在原型坐标系处理 letterbox 补边，将连续 logits 还原到原图尺寸，在对应原图框内裁剪，再按同一规则二值化。部署用 NumPy/OpenCV 重写；缩放、插值和整数舍入都要和参考路径核对。[14]

不要一条路径先二值化后缩放、另一条路径先缩放后二值化，再把差异都归为导出误差。可先通过相同后处理比较 raw 输出，把图执行误差与后处理实现误差分开。

最后叠加彩色掩膜，保证轮廓、框和原图坐标正确对齐。无目标时输出空列表与空掩膜，不访问不存在的数组元素。

### 8.7 一致性验证

选至少 30 张固定图片，覆盖横竖不同尺寸、三类目标、空背景、多目标、小目标和遮挡。

依次检查：

1. 输入张量相同，颜色、值域、布局一致。
2. PyTorch raw export 模式与 ORT 原始输出 shape、最大/平均绝对误差及相对误差。
3. 两者使用同一后处理时的框、类别和掩膜。
4. 独立后处理与官方参考处理的结果，区分插值边界差异。
5. 同类别实例按框 IoU 匹配，再比较框偏移和 mask IoU；未匹配目标单独计数。
6. 重点检查恰好靠近 conf/NMS 阈值的目标，少量目标变化需要解释，而非要求所有图逐像素绝对一致。

项目建议门槛：FP32 ORT 在固定验证集上的 mask mAP50-95 相对 PyTorch 降低不超过 **0.5 个百分点**；超出则排查，不直接接受。数值误差门槛根据输出尺度和精度设置记录，不把一个绝对误差阈值套到全部张量。

最终固定测试集统一评估并报告差异。完整 mAP 可先用 Ultralytics 验证导出的模型，检查图的精度；但该结果不自动验证你自己的后处理，独立后处理还需要上述配对检查。要宣称独立部署的 mAP，必须把该后处理的预测转换到同一评价器再计算。

**阶段产出：**ONNX、元数据、独立推理脚本、30 图一致性报告与部署截图。

**验收：**独立程序能够对图片、视频和摄像头输出正确的框与实例掩膜，精度差异有证据。

## 九、阶段 G：TensorRT 构建与独立推理

### 9.1 构建次序

1. 使用最终 FP32 ONNX 构建 FP32 engine，验证解析与推理。
2. 确认 FP32 结果正常后，构建 FP16 或允许混合精度的 engine。
3. 复用 ONNX 阶段的预处理、NMS 与掩膜后处理。
4. 先验证单图和固定图片集合，再接实时摄像头。

建议用 `trtexec` 或 TensorRT Python builder 直接从 ONNX 生成纯 engine。避免把某些框架导出的带自定义元数据头的 `.engine` 文件直接传入原生反序列化接口。

### 9.2 版本对应的精度配置

对于支持旧式精度标志的 TensorRT 10.x，可按对应版本帮助信息尝试：

```powershell
trtexec --onnx=artifacts/best_fp32.onnx --saveEngine=artifacts/best_fp32.engine
trtexec --onnx=artifacts/best_fp32.onnx --saveEngine=artifacts/best_fp16.engine --fp16
```

这些命令限定为相应的 10.x 构建路线，不能无条件用于 TensorRT 11。较新版本采用强类型网络，精度由图中的类型/转换控制；需要生成并检查合适的 FP16 图或按照对应版本的精度控制流程构建，不能继续照搬 `BuilderFlag.FP16`。[15]

因此 `scripts/build_engine.py` 应先检查 TensorRT 大版本：10.x 按支持的精度设置处理；11.x 走强类型精度路线；遇到不支持的参数明确报错。构建记录包含精度模式、输入输出 dtype、构建参数、显存和日志。

小显存设备首次可将构建 workspace 预算设置为约 1–2 GiB，再根据日志调整。workspace 是构建算法的临时内存预算，不能当作整个程序显存上限。

### 9.3 独立推理的实施步骤

`deploy/trt_backend.py` 实现：

1. 读取纯 engine 并创建 TensorRT Runtime。
2. 反序列化 engine，创建 execution context。
3. 枚举所有输入输出名称、shape 和 dtype；分割至少包括候选与原型两个输出，不能只分配一个输出。
4. 分配并复用 GPU 输入/输出内存，首版用 CUDA Python bindings。
5. 建立并复用 CUDA stream。
6. 把已预处理的输入传入 GPU，按真实 dtype 处理。
7. 用 `set_tensor_address` 绑定各输入输出地址。
8. 用对应版本的 `execute_async_v3` 等接口执行。
9. 将所有输出传回 CPU，在读取与计时之前正确同步。
10. 使用共享后处理生成结果，并释放资源。[16]

固定尺寸部署首版无需每帧重建 engine/context 或重新分配内存。目标实现不通过 Ultralytics 运行 engine；也无需使用 PyTorch执行模型。

### 9.4 精度验收

先以 TensorRT FP32 对齐 FP32 ONNX。FP16 允许更大数值变化，但要检查实例匹配、框偏移、mask IoU 和固定验证集指标。

建议 FP16 mask mAP50-95 降幅不超过 **1 个百分点**，同时单独观察三类，避免总指标掩盖某类退化。这是项目验收建议，不是 TensorRT 的通用保证。

若 FP16 不达标，先排查输入 dtype、输出读取、后处理和阈值附近实例，再考虑混合精度或部署 FP32。实际选用精度由验证结果决定。

engine 通常与平台、TensorRT 版本和 GPU 架构关联；兼容性模式有条件限制。换电脑优先带上 ONNX 与构建配置，重新生成 engine，而不是承诺一个 engine 通用于所有 NVIDIA 显卡。[3]

### 9.5 性能测试

固定同一台机器、同一模型、640×640、batch=1、相同阈值、相同后处理、相同图片或录制视频。部署期间电脑接电，尽量固定 GPU 功耗模式和后台负载。

先预热约 50 次，再用固定图片序列或录制视频测试至少 500 次。记录 mean、p50、p95；多次完整运行以检查波动。计时不包括首次模型加载和 engine 构建，启动耗时另外记录。

分两组测试：

| 测试 | 计入范围 | 目的 |
| --- | --- | --- |
| 模型执行 | 设备输入已经就绪到设备输出完成 | 比较图执行能力；GPU 需事件计时或正确同步 |
| 处理流水线 | CPU 原图 → 预处理 → H2D → 推理 → D2H → 后处理 | 比较实际使用成本，不含显示等待 |
| 摄像头演示 | 采集、推理、后处理与显示循环 | 检查观感、处理 FPS、丢帧和画面新鲜度 |

PyTorch CUDA、ORT CUDA、TensorRT 的 GPU 对比采用同一设备。ORT CPU 单独报告，不能把 CPU 与 TensorRT GPU 的差异归结为格式优化。

共享 NumPy 后处理用于公平对比。若另行实现 GPU 后处理，应作为独立优化实验说明。

测速表：

| 后端 | 精度 | 设备 | 推理均值 ms | 处理 p95 ms | 处理 FPS | mask mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | FP32 | 同一 GPU | 待测 | 待测 | 待测 | 待测 | 共享预/后处理 |
| ORT | FP32 | CPU | 待测 | 待测 | 待测 | 待测 | CPU 参考 |
| ORT CUDA | FP32 | 同一 GPU | 待测 | 待测 | 待测 | 待测 | 核实无意外回退 |
| TensorRT | FP32 | 同一 GPU | 待测 | 待测 | 待测 | 待测 | 纯 engine |
| TensorRT | FP16/混合 | 同一 GPU | 待测 | 待测 | 待测 | 待测 | 验证精度损失 |

FPS 与延迟分别报告；摄像头 30 FPS 的上限可能掩盖模型加速，因此模型/流水线速度用固定输入离线测，实时演示用于验证系统表现。

**阶段产出：**FP32 与适用的 FP16 engine、构建日志、独立 TensorRT 后端、一致性和测速报告。

**验收：**实际 GPU 推理成功；输出与 ONNX 对齐；对加速与精度的结论来自实测。

## 十、阶段 H：实时摄像头系统与稳定性

### 10.1 程序结构

先采用 OpenCV 窗口完成闭环，再根据剩余时间增加漂亮界面。主线模块职责：

| 模块 | 职责 |
| --- | --- |
| capture | 摄像头/视频读取、帧 ID、时间戳、退出 |
| preprocess | 颜色、letterbox、归一化、布局转换 |
| backend | 加载模型并输出原始候选和原型 |
| postprocess | 分数筛选、NMS、框与掩膜坐标还原 |
| render | 掩膜、框、类别、数量、耗时与 FPS |
| recorder | 截图、结果视频、运行日志 |

三个 backend 使用相同原始输出规范：统一布局、dtype 与元数据。PyTorch raw 路径以 eval/export 模式处理；官方 `predict()` 作为参考，不与已执行后处理的结果接口混用。

### 10.2 采集与推理解耦

初版可用单循环跑通。若处理速度落后于摄像头或出现明显积压，再采用：

- 采集线程持续读取，保存最新帧及采集时间。
- 推理线程只取最新可用帧，缓冲容量 1 或采用覆盖式最新帧槽。
- 未处理旧帧允许舍弃，不无限排队。
- 帧 ID、采集时间和预测结果绑定，显示对应帧；若显示更新帧而叠旧预测，会造成运动时轮廓错位。
- 文件离线评估逐帧处理，不用实时丢帧策略。

相同原则也体现在官方视频流缓冲设置中：全部排队在推理慢于采集时可能积累延迟。[5]

### 10.3 界面和日志

显示：后端名称、模型精度、实际处理 FPS、推理 ms、处理 ms、各类当前实例数、当前阈值。FPS 可用最近 30–60 帧平滑显示，但日志保留原始值。

功能：`q` 退出、`s` 保存截图、`r` 开始/停止录像，阈值通过 trackbar 或配置调整。后端首版通过启动参数选择，避免在运行中频繁加载模型。

日志：frame_id、capture_time、infer_start/end、render_time、实例数、处理时间和异常。相机读帧之后记录的时间戳只能用于“应用内帧龄”，并不等于从曝光到显示的真实延迟。若要验证真实画面延迟，可用拍摄计时器等可观察方法辅助测量，记录方法与限制。

### 10.4 性能与稳定性目标

建议先以 640 输入的 **15–30 处理 FPS** 为演示目标；实际目标在阶段 A 硬件检查和首轮测速后确认。不要在 GPU 型号未知时保证固定帧率。

处理时间若为 50 ms，单线程串行流水线理论吞吐约 20 FPS；线程重叠后不宜简单用 `1000/总延迟` 替代实测吞吐。

检查至少：

- 连续运行 20–30 分钟，内存和显存没有持续增长。
- 空画面、多目标、快速移动和遮挡时不崩溃。
- 目标框与掩膜和同一帧对应。
- 摄像头占用或断开时可读报错并释放资源。
- 截图和录像正常，退出时释放摄像头、writer 和 GPU 资源。
- 长时间运行没有不断增大的应用内帧龄。

**阶段产出：**演示程序、使用说明、短演示视频、稳定性日志。

**验收：**可以从启动到实时预测、保存结果、正常退出完成现场演示。

## 十一、代码组织和统一命令接口

以下是需要在实施中建立的文件职责与接口，不是本方案已交付的实现代码。

| 路径 | 内容 |
| --- | --- |
| `README.md` | 环境、数据、训练、部署与演示说明 |
| `requirements-train.txt` / `requirements-deploy.txt` | 两类环境依赖与版本 |
| `third_party/ultralytics/` | 锁定版本的源码与修改 |
| [configs/products_base.yaml](../configs/products_base.yaml) | 正式 20/5/5 三商品数据配置，已实现并通过实际加载验收 |
| `configs/products_expanded.yaml` | E2/E3 扩充商品训练池配置（待实现） |
| `configs/yolo11n-seg-se.yaml` | 修改模型配置 |
| `configs/runtime.yaml` | 后端、尺寸、阈值、显示参数 |
| `scripts/check_env.py` | 环境检查 |
| `scripts/annotate_product_drafts.py` | 提议轮廓、保留来源和审核状态；不直接把草稿当真值（待实现） |
| `scripts/extract_frames.py` | 视频抽帧与 group_id 记录 |
| `scripts/check_dataset.py` / `visualize_labels.py` | 标签检查与可视化 |
| [scripts/train.py](../scripts/train.py) / [evaluate.py](../scripts/evaluate.py) | 已支持显式 E1 / E1-A 配置，两份各完成 50 轮 / 150 更新及保存权重的固定 val 评价；E2/E3 与最终 test 评价待后续 |
| [scripts/analyze_product_errors.py](../scripts/analyze_product_errors.py) | 已对 E1 / E1-A 分别做 train20 / val5、低分候选、框 / 掩膜 IoU 和紧凑预览分析；不训练或推理封存 test |
| [scripts/check_augmentation_control.py](../scripts/check_augmentation_control.py) / [compare_augmentation_control.py](../scripts/compare_augmentation_control.py) | 实际加载器核对增强与标签同步；保持预算 / 数据 / 评价规则一致，对照 E1 / E1-A 并保存合并验收 |
| `scripts/export_onnx.py` / `build_engine.py` | 导出与构建 |
| `scripts/compare_backends.py` / `benchmark.py` | 一致性与性能测试 |
| `deploy/preprocess.py` / `postprocess.py` | 共享前后处理 |
| `deploy/torch_backend.py` / `onnx_backend.py` / `trt_backend.py` | 模型执行 |
| `app/webcam.py` / `render.py` | 摄像头与界面 |
| `data/desktop/images/` / `labels/` / `splits/` | 数据与冻结划分 |
| `artifacts/` | 选定权重、ONNX、engine、元数据 |
| `reports/` | 环境、数据、实验、部署与稳定性报告 |
| `demo/` | 截图、录像、展示材料 |

E1 当前可运行接口是第六节的 `--config` 路线；其余后续目标接口示例如下，其中 E2/E3 和部署脚本参数尚待实现：

```powershell
python scripts/check_env.py
python scripts/annotate_product_drafts.py --images data/raw/camera/frames --output data/raw/camera/annotations/drafts
python scripts/check_dataset.py --data configs/products_expanded.yaml
python scripts/train.py --config configs/train_baseline.yaml
python scripts/train.py --data configs/products_expanded.yaml --model yolo11n-seg.pt --name E2
python scripts/train.py --data configs/products_expanded.yaml --model configs/yolo11n-seg-se.yaml --name E3
python scripts/export_onnx.py --weights artifacts/best.pt --imgsz 640
python scripts/build_engine.py --onnx artifacts/best_fp32.onnx --precision fp32
python scripts/build_engine.py --onnx artifacts/best_fp32.onnx --precision fp16
python scripts/compare_backends.py --images data/fixed_checks --backends torch onnx trt
python scripts/benchmark.py --video demo/fixed_benchmark.mp4 --warmup 50
python -m app.webcam --backend onnx --model artifacts/best_fp32.onnx --source 0
python -m app.webcam --backend trt --model artifacts/best_fp16.engine --source 0
```

实现这些脚本时增加 argparse，与上述接口对应。E3 的权重加载必须采用第七节的正确映射；`--precision fp16` 必须根据 TensorRT 版本选择有效构建方法。

## 十二、项目验收与成果清单

### 12.0 小数据首版验收

完成指定包装在限定桌面条件下的真实微调与演示：

- 数据和轻量增强可加载，独立留出用途明确，原图 / 标签 / 配置 / 源码与权重可追溯。
- E1 权重能重载；留出图与新拍摄画面中可展示三类的类别、框和独立掩膜，保存各类成功样例及实际失败记录，不设高精度硬门槛。
- 新商品摄像头可显示原图和分割结果；最终权重能导出 ONNX 并实际加载 / 执行，保存复现命令与检查记录。
- 展示条件先限定包装正面 / 斜侧面可见、正常光照、适中距离及轻微遮挡；样本很少，指标只描述本次留出，不宣称跨包装或广泛场景泛化。
- E2、SE、TensorRT、完整独立后处理及长期稳定性按后续完整路线分别验收；未执行项不勾选。

若 E1 已满足演示需要，停止追加照片；如不足，先在累计新增 30 张以内按验证错误补样本。超出预算时先记录原因与缩小演示条件，不自动增加人工任务。

### 12.1 完整技术路线最终检查表

- [ ] 环境可复现，源码 commit、依赖与硬件信息已记录。
- [ ] 数据类别映射、标签可视化、冻结划分与分组记录完整。
- [ ] 完成 YOLO11n-seg 三类微调，保存最佳与最后权重。
- [ ] 完成至少一次场景适配对比与一次结构对比。
- [ ] SE 修改可训练、可导出，预训练权重加载经过检查。
- [ ] ONNX Runtime 独立运行，自己的框与掩膜后处理正确。
- [ ] TensorRT engine 独立运行，有精度与性能证据。
- [ ] 同一录制输入上比较后端，GPU 计时正确。
- [ ] 摄像头实时系统能稳定运行、截图、录像和退出。
- [ ] 固定测试集统一评估，结果未用于选模型或调阈值。
- [ ] README、实验表、演示视频和面试讲解齐全。

### 12.2 必须交付的内容

1. 可运行代码和依赖说明。
2. 数据准备与标注说明、冻结划分清单。
3. 基线与改进权重、配置与实验记录。
4. 最终 ONNX、适用 engine 和元数据。
5. 模型修改对照表与失败案例。
6. 后端一致性报告与测速表。
7. 摄像头演示程序与 1–2 分钟演示视频。
8. 一页项目介绍及面试口述提纲。

指标与加速倍数全部填写实测值。若某项没有完成，标为待完成；若某项没有提升，记录负结果，不能填写预期值作为结论。

## 十三、如何在假期内推进与处理阻塞

先保证每个关键接口有一条可运行路径，再加正式实验。可以交叉安排：

- 当前 M1 与 20 张试标已完成，先整理现有数据和增强预览，再做短训练；需要补拍时提供具体用途与清单。
- E1 训练时开发共享预处理、NMS 与 ORT 单图推理。
- E2/E3 训练时开发 TensorRT 后端和摄像头界面。
- engine 构建时整理环境、数据与实验记录。

| 问题 | 处理顺序 |
| --- | --- |
| GPU 训练 OOM | 先减 batch，检查缓存与异常；正式对照保持相同设置 |
| 奶盒漏检 / 草稿轮廓差 | 手工补标少量样例，统计需重画比例；先确认辅助标注是否省时 |
| 标注时间超预期 | 采用在线增强；先 E1，新增人工审核累计最多 30 张（含留出），优先少量独立场景与必要缺口；不自动增加上限 |
| 结构修改不导出 | 在正式训练前排查标准算子、通道、索引、导出模式 |
| 修改没有改善 | 保存对照与解释，按验证集选择原基线用于最终产品 |
| ONNX 框/掩膜偏移 | 检查补边、实际缩放、坐标系和插值顺序 |
| ORT CUDA 无法加载 | 核对驱动、CUDA/cuDNN、包版本与 DLL；CPU 路径维持开发进度 |
| TensorRT 构建失败 | 查看 parser 日志；回到固定形状 FP32，核对版本与精度 API |
| 实时延迟持续增加 | 最新帧缓冲、减小摄像头缓冲、检查显示与后处理瓶颈 |
| FPS 没提升 | 先分段计时，确认是否受采集上限或 CPU 后处理限制 |

TensorRT 因硬件或软件确实不能运行时，仍完成其余模块并明确记录阻塞。单纯导出文件但没有执行和验证，不能计为 TensorRT 部署完成。

INT8、动态尺寸、P2 分支、复杂界面、C++、服务化与多摄像头可延后。核心主线不以这些拓展为验收条件。

## 十四、面试讲解路线

按“问题 → 数据 → 模型 → 优化 → 部署 → 结果”讲述。当前 20 张人工数据试标已完成；以下训练、修改与正式部署为完成后的目标路线，未执行项用计划表述，首版先讲真实数据、轻量增强、E1 与摄像头 / ONNX，结构对照另按实际完成情况讲述：

1. **任务：**识别两种指定奶盒和一种瑞幸杯的可见包装，输出每个实例的类别、位置、轮廓与当前数量。
2. **数据：**已有 20 张人工审核原图；训练使用在线增强、标签同步变换，按拍摄组划分，新增真实样本和人工审核限定在累计 30 张内。独立留出及各类实例数量按实际记录。
3. **训练：**用 YOLO11n-seg 迁移学习，分别评估框与掩膜，针对真实场景错误做数据适配。
4. **修改：**在 P3 输入分支增加 SE，控制数据和训练设置，记录加载、训练、导出与对照结果。
5. **部署：**导出 ONNX，独立实现预处理、NMS 和原型掩膜还原，再用 TensorRT 构建并运行 engine。
6. **工程：**以相同输入比较精度与速度，处理帧积压、坐标还原和连续运行稳定性。
7. **结果：**填写真实指标、实测帧率与限制；指出修改是否值得部署。

可能追问：为什么实例分割而不是分类；如何避免数据泄漏；为什么用预训练；注意力是否一定有效；ONNX 为什么可以脱离 PyTorch执行；TensorRT 为什么快；FP16 的代价；为什么模型很快而视频仍延迟；为什么不同后端掩膜边缘不同；为什么不能把一个 engine 直接发到任意电脑。

项目价值来自你能完整解释并复现这条链路。已有模块组合可以称为结构调整或工程优化；只有实验与方法支持时才进一步讨论创新性。

## 十五、官方资料与学习顺序

以下为核对过的官方文档、官方源码或原论文。网页内容与 API 可能更新，执行时以所锁定的软件版本为准。

1. [Ultralytics YOLO11 模型](https://docs.ultralytics.com/models/yolo11/)
2. [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)
3. [TensorRT 支持矩阵与 engine 兼容性](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
4. [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
5. [Ultralytics 摄像头与视频推理](https://docs.ultralytics.com/modes/predict/)
6. [YOLO11-seg 官方 YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml)
7. [COCO 数据集](https://cocodataset.org/#download)
8. [实例分割数据格式与 COCO 转换](https://docs.ultralytics.com/datasets/segment/)
9. [Ultralytics 训练参数](https://docs.ultralytics.com/modes/train/)
10. [Ultralytics 实例分割任务](https://docs.ultralytics.com/tasks/segment/)
11. [Squeeze-and-Excitation Networks 原论文](https://arxiv.org/abs/1709.01507)
12. [模型 YAML 与自定义模块指南](https://docs.ultralytics.com/guides/model-yaml-config/)
13. [模型导出参数](https://docs.ultralytics.com/modes/export/)
14. [掩膜与坐标后处理源码说明](https://docs.ultralytics.com/reference/utils/ops/)
15. [TensorRT 11 移除的精度 API 与替代方法](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-reference.html)；[trtexec 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)
16. [TensorRT Python API 与推理流程](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)
17. [Ultralytics 模型辅助标注](https://docs.ultralytics.com/datasets/segment/#auto-annotation)；当前草稿导出以锁定的 [Results.save_txt 源码](../third_party/ultralytics/ultralytics/engine/results.py)为准

按“模型与摄像头 → 数据与训练 → 配置与结构修改 → 输出与掩膜 → ONNX 与 TensorRT”阅读即可，不需要一次读完全部文档。
