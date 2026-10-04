# M5：商品 ONNX 独立部署与接口说明

更新：2026-10-04。**M5-01～06 全部完成，M5 6/6，主线 30/54，完整模块 4/9。** B1 已完成 ONNX 导出、共享前后处理、ORT CPU / CUDA 图片视频程序、30 固定输入配对及 val5 同条件部署质量评价。三个后端的框 / 掩膜 mAP50-95 均为 0.8205 / 0.7434，未测到部署降幅；104 项工程测试、261 个历史文件保护通过。0.25 展示门槛检出 7/8、瑞幸杯漏检保留；0.001 评价门槛下 CPU 有一组低分候选的裁剪边界差异，已诊断，不宣称所有低分掩膜完全一致。摄像头仍用 PyTorch、test5 封存，无新训练或正式 FPS。正式后续为 M6-01 TensorRT 构建条件；也可按展示需求先接摄像头 ORT。见[文件命令](#m5-04-file-inference)、[固定输入配对](#m5-05-backend-parity)及[本次质量与可视化](#m5-06-validation-quality)。

<a id="m5-01-export"></a>

## 1. M5-01 完成了什么

这一步把已训练好的商品模型转换成 ONNX 计算图，并明确后续程序应传入什么、会收到什么。它不改变模型结构、不重新训练，也不计算新的 mAP。

| 项目 | 实际验收结果 |
| --- | --- |
| 模型来源 | B1=`products-v1_E1A_seed42_B1`，E1-A 第 50 轮 best，原权重路径 / SHA 与冻结记录一致 |
| 类别顺序 | 0=`sam_whole_milk`，1=`yili_shuhua`，2=`luckin_cup`；真实三类头，没有 COCO 80 类输出 |
| 导出选项 | imgsz640、batch1、dynamic=false、quantize=32、opset17、simplify=false、nms=false、device0 |
| 图检查 | `onnx.checker.check_model(full_check=True)` 通过；IR=8、opset=17、393 节点、无 NonMaxSuppression 节点 |
| 保存模型 | [best_fp32.onnx](../artifacts/B1/best_fp32.onnx)，11,584,852 bytes，约 11.05 MiB |
| ONNX SHA256 | `35bcd6f00c04299fd9846b746a1488fc48ea9e84877e4009434f31baef5e28d0` |
| 运行检查 | 同一保存输入在 PyTorch CUDA FP32 导出头、ORT CPU、ORT CUDA 执行，三张 val 的两个输出全部通过容差 |
| 实际 CUDA 证据 | ORT profile 中 795 次 CUDA 节点执行事件，未发现 CPU 节点执行事件；事件次数不是模型节点数或 FPS |
| 工程检查 | 65 项测试通过；新增 4 项接口 / 封存数据 / 数值检查，原 61 项保留 |
| 输入保护 | 219 个输入文件核对一致；包含原 B1 的 216 个文件及 B1 清单 / 验收 / 本次配置 |

采用 Python3.9.23、Ultralytics8.4.171 / 锁定 commit、torch2.8.0+cu129、onnx1.17.0、onnxruntime-gpu1.19.2，GPU RTX5060Ti。没有修改或安装环境依赖；上游源码保持 clean。导出时只修改加载对象的内存中 `pt_path` 属性来指定 ONNX 目标位置，原 best.pt 未覆盖，也没有另存一份 PT 权重。日志显示的 `artifacts/B1/best_fp32.pt` 是导出命名参照，并非生成的新 PT 文件。

## 2. 输入与输出到底是什么

| 接口 | 名称 / shape / dtype | 解释 |
| --- | --- | --- |
| 输入 | `images [1,3,640,640] float32` | 一张 RGB 图片，NCHW 排列，像素归一化到 0～1 |
| 候选输出 | `output0 [1,39,8400] float32` | 8400 个候选位置，每个包含 4 个框坐标、3 个类别分数、32 个掩膜系数 |
| 原型输出 | `output1 [1,32,160,160] float32` | 32 张共享的掩膜基底，尚不是每个目标的最终掩膜 |

`output0[0, :, i]` 表示第 i 个候选的 39 个数。通道区间采用左闭右开：

```text
0:4   → cx, cy, w, h：补边后的 640×640 输入坐标，单位为像素
4:7   → 山姆 / 伊利 / 瑞幸三个类别分数，已经经过 sigmoid
7:39  → 32 个掩膜系数，没有 sigmoid
```

没有额外 objectness 通道，也不需要再把类别分数经过 sigmoid。8400 是候选位置数，不是画面里物品数量；DFL 的框解码已在计算图里完成，后续仍需把 xywh 转为 xyxy、筛选与 NMS。

原型也没有单独做 sigmoid。保留实例的 32 个系数与展平的 32×25600 原型矩阵相乘，得到 160×160 掩膜 logits。最终实例掩膜还需插值、去补边、框裁剪和二值化；这些操作不在本次 ONNX 图内。

## 3. 前后处理接口约定（M5-02、03 已实现）

**M5-02 预处理约定：**原图 BGR / uint8 / HWC → 保持比例缩放 → 居中填充到 640×640，填充值 114 → RGB → float32 / 255 → NCHW / 连续内存。首版 `auto=false`、`scale_fill=false`、`scaleup=true`，使用 OpenCV 双线性缩放。必须保留原图尺寸、缩放比例和上下左右整数补边，供框与 mask 还原。

锁定 LetterBox 的比例为 `r=min(640/h,640/w)`，缩放宽高分别为 `round(w*r)`、`round(h*r)`；单边补量采用 `round(half_pad-0.1)` 与 `round(half_pad+0.1)`，处理奇数补边。M5-01 当时只用官方 LetterBox 生成参考；本次 M5-02 已完成独立实现并与参考逐像素核对，见[第 6 节](#m5-02-preprocess)。导出元数据中的 implementation pending 是 M5-01 的历史状态，保持原文件不变；当前实现状态由 M5-02 独立验收记录承接。

**M5-03 后处理约定：**展示 conf=0.25、指标计算 conf=0.001、逐类别 NMS IoU=0.7、max_det=300、三类全部保留。普通展示每个候选取最高类别分数；评价器的 multi-label 设置需单独匹配参考规则。保留候选索引，让框和它的 32 个系数一起经历筛选 / NMS。

首版参考当前 `retina_masks=true` 路径：框恢复原图坐标；系数与原型组合出 logits；按原图比例去掉掩膜补边并双线性插值，按 logits>0 二值化，再用原图框裁剪。logits>0 等价于 sigmoid(logits)>0.5，但锁定实现直接比较 logits，不对每张原型先 sigmoid。最后与官方预测器一致，丢弃全空掩膜对应的检测。空检测也必须正常返回。

本轮依据的是锁定源码的 `LetterBox`、`Detect/Segment`、NMS、`process_mask_native` 与 `SegmentationPredictor`，不是从其他版本照搬规则。具体接口参数、通道含义和源码来源也保存在机器元数据。

## 4. 实际执行与数值对照

仅使用已有 val 的 001、004、005：两个商品场景及一个无目标场景。不使用封存 test、不新增拍摄、不复制源图。三张 720×1280 原图按固定方形路线得到 640×640 输入，本轮不计算目标数量或识别准确率。

PyTorch 参考重新加载 B1，融合卷积 / BN 并使用相同导出头模式，FP32、TF32=false；ORT CPU 和 CUDA 接收同一保存张量。检查标准为逐元素：

```text
abs(ORT - PyTorch) <= 0.001 + 0.0001 * abs(PyTorch)
```

| 执行后端 | 三图候选输出最大绝对误差 | 三图原型输出最大绝对误差 | 容差外元素 |
| --- | --- | --- | --- |
| ORT CPU | 0.00079346 | 0.00001812 | 0 |
| ORT CUDA，use_tf32=0 | 0.00039673 | 0.00001145 | 0 |

输出 shape、float32 dtype、有限值均通过。CUDA 会话实际使用 CUDAExecutionProvider，并用 profile 确认节点执行；不能只凭 provider 名称声称 GPU 运行。原始输出对照通过不代表完整分割结果或部署 mAP 已验收，微小候选差异在阈值 / NMS 边界的影响留到 M5-05 检查。

B1 的历史 val / 摄像头使用矩形补边，本轮采用固定方形输入；M5-06 要在相同方形输入与评价规则下取得 PyTorch 参考，不能直接将历史 0.7807 当部署降幅参照。CPU / CUDA 这里没有预热与正式计时，不生成加速倍数或 FPS 结论。

## 5. 文件、复核和下一步

| 文件 | 用途 |
| --- | --- |
| [export_products.yaml](../configs/export_products.yaml) | B1 来源、导出参数、三张 val ID、输出位置与容差 |
| [export_onnx.py](../scripts/export_onnx.py) | 导出入口、同输入 raw 检查、元数据与验收；已有输出拒绝覆盖 |
| [onnx_contract.py](../deploy/onnx_contract.py) | 图接口 / 类别 / NMS 检查、val 来源限制、数值比较 |
| [test_onnx_export.py](../tests/test_onnx_export.py) | 4 项新增测试，拒绝错误模型接口、类别映射、封存数据和非有限输出 |
| [best_fp32.metadata.json](../artifacts/B1/best_fp32.metadata.json) | 模型 SHA、B1 权重来源、输入输出、类别、预后处理约定、参考图片来源 |
| [export_reference.npz](../artifacts/B1/export_reference.npz) | 三图同输入与 PyTorch 导出头输出，集中保存一份，供后续定位数值差异 |
| [M5-01_B1_onnx_export.json](../reports/deployment/M5-01_B1_onnx_export.json) | 机器验收：图检查、CPU / CUDA 对照、逐图误差、profile SHA、输入保护与范围 |
| [导出日志](../logs/deployment/M5-01_B1_onnx_export.log) | 当次导出与运行信息 |

```powershell
conda activate yolo
# 已完成首次导出，现在使用只读复核，不重新导出或推理。
python scripts/export_onnx.py --check
# 原 B1 数据 / 权重 / 配置仍可独立复核。
python scripts/freeze_product_baseline.py --check
```

首次导出命令是 `python scripts/export_onnx.py`，当前不要重复执行；入口拒绝覆盖已有模型 / 元数据 / 参考数组 / 报告 / 日志。要做新的导出版本须另用新输出配置。`--check` 核对已有模型、图接口、文件哈希和来源，不执行模型或写文件。

ONNX / 数组 / profile 保存在已有忽略规则下的 `artifacts/B1/`，配置、代码、接口说明与小型验收记录供 Git 展示。没有新增模型 PT 副本，也不保存大量图片预览。本次新照片、训练、优化器更新和最终测试推理均为 0；当前摄像头仍使用已验收的 PyTorch 入口。

共享前后处理、统一后端及完整 ORT 文件推理程序已接续完成，见第 6～8 节。现有摄像头仍使用原 PyTorch 入口，摄像头多后端接入按 M7 后续任务开展。

<a id="m5-02-preprocess"></a>

## 6. M5-02：共享预处理实现与验收

这一环节把摄像头或本地图片的原始像素转换为模型输入，同时记录画面缩放和补边，供后续恢复框、掩膜。不会改变权重或识别能力；本次不执行模型，不计算新的 mAP。

```text
BGR uint8 [H,W,3]
→ 按比例缩放、居中补 114 到 640×640
→ BGR 转 RGB、float32 / 255、NCHW 连续内存
→ images [1,3,640,640] + LetterboxGeometry
```

| 产出 | 用途 |
| --- | --- |
| [deploy/preprocess.py](../deploy/preprocess.py) | `preprocess_bgr(image)` 返回张量与几何信息；只依赖 NumPy/OpenCV，拒绝错误图片格式；`validate_preprocessing_contract(metadata)` 核对 B1 静态输入约定 |
| [scripts/check_preprocess.py](../scripts/check_preprocess.py) | 与锁定官方 `BasePredictor.preprocess` / `LetterBox.get_params` 对照；首次保存记录，已有记录拒绝覆盖，`--check` 只读复核 |
| [tests/test_preprocess.py](../tests/test_preprocess.py) | 5 项新增测试：颜色 / 归一化、横竖 / 放大、奇数补边 / 比例、非连续只读输入、错误图片 / 不兼容接口 |
| [M5-02_B1_preprocess.json](../reports/deployment/M5-02_B1_preprocess.json) | 每个输入的几何参数、逐像素一致性、保存输入 SHA、独立导入与 228 个历史文件保护 |

本次 5 张 val 原图加 12 个内存生成的尺寸测试输入，输出像素和几何参数全部相同，最大绝对误差 **0**。尺寸测试覆盖横图、竖图、方图、奇数补边、放大、小尺寸及非连续只读数组；这些内存图案没有保存为数据集，也不构成新增独立评价。M5-01 三份保存输入逐像素和 SHA 均一致，无需重复模型推理。

新进程仅导入本模块后，`torch` / `ultralytics` 均未导入。70 项全项目测试通过；228 个受保护历史文件的 SHA 和修改时间保持不变，B1 / M5-01 复核通过，锁定源码及环境无漂移。此次只新增三份代码和一份小型 JSON，更新既有文档；没有新增照片、标注、训练、权重、输入数组副本或最终 test 推理。

### 几何信息怎样读

所有 `shape_hw` 都是 **高、宽**；`ratio_xy` 是 **横、纵比例**；`padding_ltrb` 是 **左、上、右、下**。

| 字段 | 原图高 720、宽 1280 的实际结果 |
| --- | --- |
| `original_shape_hw` | `[720,1280]` |
| `input_shape_hw` | `[640,640]` |
| `resized_shape_hw` | `[360,640]`，补边之前的实际尺寸 |
| `ratio_xy` | `[0.5,0.5]`，官方采用的理想缩放比例 |
| `padding_ltrb` | `[0,140,0,140]` |

高 481、宽 640 的输入会补上 79 / 下 80；高 640、宽 481 会补左 79 / 右 80。保留整数补边可避免以后裁掉灰边时差一个像素。取整后的实际缩放尺寸也单独保存，不能用其宽高各自反算的比例替换官方 `r`。本环节只提供几何信息，框和掩膜的还原代码在 M5-03 实现并核对。

### 使用与复核

```python
from deploy.preprocess import preprocess_bgr, validate_preprocessing_contract

# metadata 是读取的 best_fp32.metadata.json，加载后端时核对一次。
validate_preprocessing_contract(metadata)
images, geometry = preprocess_bgr(frame_bgr)
# images 已是 RGB、float32、/255、NCHW，不再重复转换或归一化。
# 后续 ORT 接口：session.run(None, {"images": images})
# 后续 PyTorch raw 接口：torch.from_numpy(images).to(device)
```

上面的后端调用保留 M5-02 当时的用法示例；接续完整统一后端已在第 8 节实现。可以直接运行本阶段只读验收：

```powershell
conda activate yolo
python scripts/check_preprocess.py --check
```

接续 M5-03～06 已完成，详见第 7～10 节；商品 ONNX 模块已收尾，当前无需补拍或重新标注。

<a id="m5-03-postprocess"></a>

## 7. M5-03：独立框与掩膜后处理

这一环节把模型 raw 输出转成原图上的商品实例，形成完整的后处理接口：

```text
output0 [1,39,8400]
→ 最高类别分数 > 0.25 → xywh 转 xyxy → 逐类 NMS / IoU=0.7
→ 保留候选索引、类别、分数和对应的 32 个系数 → 框去补边 / 除比例 / 裁到原图范围

对应系数 [N,32] × output1 原型 [32,25600]
→ [N,160,160] logits → 去原型尺度的补边 → 双线性插值到原图
→ logits > 0 → 按原图框裁剪 → 丢弃全空掩膜 → 商品实例
```

本项目模型本身已输出三个商品类别，普通展示取每个候选分数最高的类别。类别分数已经经过 sigmoid，不再计算 sigmoid 或乘一个不存在的 objectness。逐类 NMS 只去除同一类别的重叠候选；不同类别可以保留相同位置的框。候选索引始终跟随框、分数与掩膜系数，不能筛完框之后拿筛选前的系数组合掩膜。

### 文件与输出接口

| 文件 | 用途 |
| --- | --- |
| [deploy/postprocess.py](../deploy/postprocess.py) | 仅 NumPy/OpenCV：分数筛选、逐类 NMS、框还原、原型组合、掩膜还原和空掩膜筛除；核对 B1 元数据约定 |
| [scripts/check_postprocess.py](../scripts/check_postprocess.py) | 复用 M5-01 保存 raw 输出，逐步对照锁定官方函数；拒绝覆盖已有验收，支持 `--check` 只读复核 |
| [tests/test_postprocess.py](../tests/test_postprocess.py) | 10 项新增测试：类间 NMS、系数绑定、严格阈值、同分顺序、多标签、空结果、max_det 顺序、奇数补边、浮点框裁剪、错误输入等 |
| [M5-03_B1_postprocess.json](../reports/deployment/M5-03_B1_postprocess.json) | 集中保存 18 组逐步对照、最终框与索引、掩膜一致性、来源 SHA 和历史文件保护 |

```python
from deploy.postprocess import postprocess_b1, validate_postprocessing_contract

validate_postprocessing_contract(metadata)  # 后端初始化时检查一次
result = postprocess_b1(output0, output1, geometry)  # geometry 来自 preprocess_bgr
```

`result` 每一行对应同一个实例：`boxes_xyxy [N,4]` 是原图像素坐标，`class_ids [N]` / `scores [N]` 是类别 / 分数，`masks [N,H,W]` 是原图大小的 uint8 0/1 掩膜；另有 `candidate_indices [N]`、`mask_coefficients [N,32]` 和筛选计数。无目标时返回 `N=0` 的完整形状，不返回错误。框坐标仍保留小数，裁剪掩膜依据 `x1<=x<x2, y1<=y<y2`，不先把框取整。

### 验收结果

复用 val001 / 004 / 005 的三组已保存 raw 输出，每组分别检查展示 `conf=0.25, multi_label=false` 和评价筛选规则 `conf=0.001, multi_label=true`，共 6 组；加 12 组内存构造的横竖 / 奇数 / 放大、重复框、多标签、无候选、空掩膜及 max_det 边界输入，共 **18 组**。参考是锁定源码 CPU FP32 的 NMS、`scale_boxes`、`process_mask_native`、`scale_masks`；三个展示场景额外核对 `SegmentationPredictor.construct_result`，没有模型前向或重新导出。

| 对照项目 | 本次实测 |
| --- | --- |
| NMS 后候选索引 / 类别 / 分数 / 掩膜系数 | 全部一致 |
| 原图框坐标最大绝对误差 | 0 |
| 插值后 logits 最大绝对误差 | 0.0000076294 |
| 二值掩膜 IoU | 所有非空对应实例均为 1.0 |
| 二值掩膜不同像素 | 0 |
| 全项目工程测试 | 80 项通过，新增 10 项 |
| 原数据 / 权重 / M5-01、02 等历史文件 | 232 个文件 SHA / 修改时间一致 |
| 独立新进程 | 后处理不导入 torch / ultralytics，正常生成原图框和掩膜 |

这里的 mask IoU 是**独立实现与官方后处理的掩膜比较**，不是与人工标签比较，不表示识别准确率为 100%。工程检查预先设定框误差 ≤1e-4，logits 容差 atol=2e-4 / rtol=2e-5，单实例 mask IoU≥0.999、不同像素占原图比例≤1e-5；本次二值结果完全相同。OpenCV 与 PyTorch 双线性插值的浮点舍入并非普遍逐位相同，后续仍在固定输入集复核。

展示模式下保存输出的实际筛选计数：

| 保存 val 输入 | 原始候选位置 | 分数 >0.25 | NMS 后 | 原图实例 / 掩膜 |
| --- | --- | --- | --- | --- |
| 001 | 8400 | 14 | 3 | 3 |
| 004 | 8400 | 17 | 3 | 3 |
| 005，无目标背景 | 8400 | 0 | 0 | 0 |

这些是预测计数，不是新质量评价。低门槛多标签检查得到 233 / 223 / 203 个最终候选，包含许多低分预测；只用于验证评价前的筛选规则，不用它作为展示门槛，也未计算 mAP。

### 实现细节与复核

原型只有 160×160，去补边时按这个尺度和原图比例重新计算取整规则；不能简单把 640 输入的整数补边除以 4。先对 logits 插值、按 `>0` 二值化，最后裁到原图框；全空掩膜筛除发生在 max_det 截断之后，不再从后面的候选补足数量。中间浮点掩膜按块处理，输出仍需 `N×H×W` 字节存储。

同分候选采用原候选顺序稳定处理；官方 NMS 的同分顺序可能随后端而变化，特别是在 max_det 截断边界。当前保存样本的候选一致，另有同分稳定性测试；未来固定配对检查需识别这类排序歧义。M5-01 元数据中的后处理 pending 保留为导出时的历史状态，当前完成状态由 M5-03 验收承接。

```powershell
conda activate yolo
python scripts/check_postprocess.py --check
```

本次新增三份代码和一份集中 JSON，更新原文档；新照片、标注修改、训练、模型推理、最终 test 推理和保存数组副本均为 0。摄像头仍使用原 PyTorch 入口，本次没有建立完整 ORT 图片 / 视频应用或评价新 mAP。

接续 M5-04 已完成完整文件推理和参考后端，见下节。

<a id="m5-04-file-inference"></a>

## 8. M5-04：完整独立图片 / 视频程序

现在可以直接从本地文件得到三种商品的类别、分数、原图框和掩膜：

```text
图片 / 视频帧 → 同一份 preprocess_bgr → ORT CPU / ORT CUDA / PyTorch raw
            → 同一份 postprocess_b1 → InferenceResult → 标注图片 / 视频
```

### 已实现的文件

| 文件 | 职责 |
| --- | --- |
| [infer_products.yaml](../configs/infer_products.yaml) | 固定 B1 模型来源、metadata、默认 ORT / CUDA0、FP32、conf=0.25 和 NMS 参数 |
| [base_backend.py](../deploy/base_backend.py) | 核对 B1 / 导出来源与资产 SHA，统一预处理 → raw 执行 → 后处理；按需加载具体后端 |
| [results.py](../deploy/results.py) | 两类后端共用 `InferenceResult`，含实例、原图几何、类别映射、后端信息和分段调用耗时 |
| [onnx_backend.py](../deploy/onnx_backend.py) | ORT 静态接口、CPU / CUDA provider、Windows DLL 搜索和可选 profile；不导入 torch / ultralytics |
| [torch_backend.py](../deploy/torch_backend.py) | 加载 B1 原权重，采用与导出一致的融合 / export-head FP32 路径；直接网络前向，不调用 `YOLO.predict` |
| [infer_products.py](../app/infer_products.py) | 图片 / 文件视频入口、框和掩膜绘制、结果保存、可选显示、Q/Esc 退出和资源释放 |
| [check_product_deployment.py](../scripts/check_product_deployment.py) | 新进程实际后端 / CLI 验收、同保存输入 raw 与完整结果对照、CUDA profile 及输出解码；已有验收只读复核 |
| [test_product_deployment.py](../tests/test_product_deployment.py) | 8 项新增工程测试：统一接口、错接口、模型 / TF32 配置漂移、封存图片、拒绝覆盖、逐帧处理 / 异常释放、CPU 回退拦截 |

每帧只执行一次 `backend.predict`，框、类别、掩膜都属于当前帧，保存与显示复用同一张标注画面。`InferenceResult.instances` 沿用 M5-03 的实例结构，`timings_ms` 记录预处理、raw 执行（含传输完成）、后处理和核心总耗时；不包含解码、绘图、写文件或模型初始化，不作为正式 FPS。CLI 的总耗时另包含模型启动与文件处理。

### 本次实际验证

三个独立子进程各使用原 val001 / 004 / 005，检查同保存输入 raw 输出，再用真实原图执行完整共享流程。raw 容差仍为 atol=0.001 / rtol=0.0001；完整实例要求候选 / 类别一致、原图框误差≤0.01 像素、分数误差≤1e-4、对应 mask IoU≥0.999。

| 后端 | 三图实例数 | 原图框最大误差，像素 | 对应 mask IoU 最低 | 导入 torch / ultralytics |
| --- | --- | --- | --- | --- |
| ORT CPU | 3 / 3 / 0 | 0.00006104 | 1.0 | 均否 |
| ORT CUDA | 3 / 3 / 0 | 0.00012207 | 1.0 | 均否 |
| PyTorch CUDA raw 参考 | 3 / 3 / 0 | 0 | 1.0 | 均是，作为参考后端 |

ORT CUDA profile 记录 **1590 次 CUDA 节点执行事件、0 次 CPU 节点执行事件**，来自三张输入各一次 raw 对照和一次完整推理。provider 列表本身不作为 GPU 执行证据；当前入口请求 CUDA 而会话实际回退为 CPU 时直接报错。事件次数不是模型节点数或 FPS。

实际 CLI 在新进程运行了 CPU 图片、CUDA 图片、CPU 文件视频，均不导入训练框架。两份 PNG 标注图逐字节相同，尺寸 1280×720；三帧输入 / 输出视频均可重新解码成三帧同尺寸画面，正常到文件末尾退出，捕获器 / 写入器 / 后端释放通过。**88 项全项目测试通过；236 个历史文件 SHA / 修改时间一致**，原图、人工标签、训练权重、既有前后处理、M5-01～03 和源码 / 环境保持不变。

| 可直接查看的结果 | 内容 |
| --- | --- |
| [ORT CPU 图片](../demo/deployment/B1/M5-04_onnx_cpu.png) | 三商品框、类别、分数与原图掩膜 |
| [ORT CUDA 图片](../demo/deployment/B1/M5-04_onnx_cuda.png) | GPU 实际完整推理，与 CPU 标注图相同 |
| [输入视频片段](../demo/deployment/B1/M5-04_input_fixture.avi) | 三张已存在 val 图按 5 FPS 编码成短文件，仅验证视频处理，不是新拍摄或新的独立数据 |
| [分割结果视频](../demo/deployment/B1/M5-04_onnx_cpu.avi) | 三帧 ORT CPU 逐帧结果，实例数 2 / 3 / 0 |
| [M5-04_B1_file_inference.json](../reports/deployment/M5-04_B1_file_inference.json) | 来源、raw / 实例比较、真实 CLI 输出、provider 事件、解码 / 释放、SHA 与任务进度 |

视频第一帧少检伊利：原 PNG 的该类最高分约 **0.250130**，MJPG 编码再解码后约 **0.232576**，低于相同 0.25 门槛。原图和压缩视频像素不同，不能要求两者预测数相同；本次记录这一模型临界分数 / 输入压缩问题，不调低门槛或覆盖 B1。仅凭这组读写检查不能判断真实视频准确率、长期稳定性或新场景泛化。这里的 mask IoU 仍是与保存参考比较，不是与人工真值比较。

### 如何使用

从项目根目录运行；`--output` 需要新路径，已有文件会被拒绝覆盖。默认 `onnx / cuda:0`，也可显式使用 CPU 或参考后端。

```powershell
conda activate yolo
# 默认 ORT CUDA：换成你自己的图片路径也可以。
python app/infer_products.py --source data/raw/camera/frames/holdout_val_20261004_095145/holdout_val_20261004_095145_001.png --output demo/deployment/B1/my_image_cuda.png
# ORT CPU，整个进程不导入训练框架。
python app/infer_products.py --backend onnx --device cpu --source data/raw/camera/frames/holdout_val_20261004_095145/holdout_val_20261004_095145_001.png --output demo/deployment/B1/my_image_cpu.png
# 同前后处理的 PyTorch raw 参考路径。
python app/infer_products.py --backend torch --device cuda:0 --source data/raw/camera/frames/holdout_val_20261004_095145/holdout_val_20261004_095145_001.png --output demo/deployment/B1/my_image_torch.png
# 文件视频，AVI / MP4 输出均有入口；本次实测输出为 AVI。
python app/infer_products.py --backend onnx --device cpu --source demo/deployment/B1/M5-04_input_fixture.avi --output demo/deployment/B1/my_video.avi
# 只读核对当前验收，不重新执行模型或增加产物。
python scripts/check_product_deployment.py --check
```

添加 `--show` 可显示结果；图片按任意键关闭，视频按 Q / Esc 退出；`--max-frames 100` 可限制文件视频帧数。当前验收自动保存文件，没有实际打开新窗口；真实摄像头仍使用 `app/live_camera.py --config configs/live_products.yaml`，未切换到 ORT。正式封存 test 原图、配对图片及字节相同副本由文件入口拦截，开发输入使用 train / val 或自行提供的非测试素材。

Windows 的 ORT1.19.2 会通过进程内 PATH 和 DLL 搜索目录找到现有 CUDA12 / cuDNN9。当前优先使用 yolo 环境 `torch/lib` 中已有 DLL，不导入 torch Python 包；没有修改系统 PATH、安装新包或下载新模型。本次在现有本机环境验收，独立发行包 / 干净移机尚未验证。固定清单与来源文件仍供入口核对。

M5-04 新增 8 份代码 / 配置、一份集中 JSON，示例图片 / 视频 / profile 在已有忽略目录中；没有新照片、标注、训练、权重副本、ONNX 再导出或最终 test 推理。当时下一步为 M5-05，接续固定输入配对已完成，见下节。

<a id="m5-05-backend-parity"></a>

## 9. M5-05：30 个固定开发输入的后端一致性检查

本环节验证同一模型从 PyTorch 换到 ONNX 后，原始输出和最终框 / 掩膜是否保持一致。输入清单、变换、源图 SHA、实际归一化 tensor SHA 和容差均在推理前固定，没有根据检查结果放宽阈值。

### 输入与新增文件

| 文件 | 职责 |
| --- | --- |
| [parity_products_B1.json](../configs/parity_products_B1.json) | 固定 30 个输入的来源、用途、内存变换、原图 / tensor SHA 和比较规则 |
| [parity.py](../deploy/parity.py) | 同类别框 IoU 匹配、框 / 分数 / mask 差异、未匹配实例、候选与系数绑定及同分记录 |
| [check_backend_parity.py](../scripts/check_backend_parity.py) | 隔离进程实际推理、同 raw 官方后处理、GPU profile、临时数组清理和只读复核 |
| [test_backend_parity.py](../tests/test_backend_parity.py) | 8 项防错测试：输出换序、缺失 / 错类、框 / 分数 / 掩膜容差、空结果、绑定、封存数据与输入漂移 |
| [M5-05_B1_backend_parity.json](../reports/deployment/M5-05_B1_backend_parity.json) | 逐输入、逐后端和汇总验收；没有额外逐图记录目录 |

输入由原 train20 / val5 共 **25 张真实原图**与 **5 个工程变体**组成：361×641 奇数横图、顺时针旋转 90° 的竖图、321×321 方图、1001×333 竖向拉伸图、17×29 小尺寸负样本。尺寸按 H×W 记录。所有变体只在内存生成，不新增照片、标注或训练数据；拉伸是检查几何处理的输入，不代表真实独立场景。共覆盖 5 张原始无目标图及其 1 个小尺寸变体，最终 test5 没有参与推理。

### 实际比较与结果

三个独立运行进程分别执行 PyTorch CUDA、ORT CPU、ORT CUDA，各 30 次前向。先以完全相同的 RGB / FP32 / NCHW tensor 比较两个 raw 输出，再让两个后端共用独立后处理，最后对所有 90 份 raw 分别核对锁定官方 NMS、native 掩膜和最终结果构造。

实例按同类别、原图框 IoU≥0.9 做一对一匹配，再要求所有实例匹配、框最大偏差≤0.01 像素、分数偏差≤1e-4、mask IoU≥0.999；没有实例的双空结果记为空对照，mask IoU 留空，不伪造测量值。候选 / 32 个系数的绑定另行核对。

| 比较 | 组数 | 匹配实例数 | 双空组数 | 框最大偏差（像素） | 最低对应 mask IoU | 未匹配实例 |
| --- | --- | --- | --- | --- | --- | --- |
| ORT CPU / CUDA 分别对 PyTorch，共享后处理 | 60 | 88 | 16 | 0.0001220703 | 1.0 | 0 |
| 三后端 raw 的独立后处理对官方后处理 | 90 | 132 | 24 | 0 | 1.0 | 0 |

两项比较的二值掩膜差异像素均为 0，候选绑定差异均为 0。后端分数最大偏差为 5.3644e-7。两个 raw 输出共 120 项比较，全部满足 `abs(actual-reference) ≤ 0.001 + 0.0001 × abs(reference)`；候选输出最大绝对误差为 0.00120544，原型为 2.4796e-5。这里使用绝对和相对容差之和，不能把单独的 0.001 当成所有数值的绝对上限。

ORT CPU / CUDA 进程均未导入 torch / ultralytics，CUDA profile 确认 **7950 次 GPU 节点执行事件、0 次 CPU 节点执行事件**。每条路线有 8 个空输出输入；其中 c23 是有目标的验证原图，c29 是有目标原图的竖向拉伸变体，三个后端共同输出为空。这些模型弱点保留，部署一致不能证明识别正确或提高了准确率。本批真实推理没有发现超过 0.25 门槛的精确同分候选；同分排序仍受官方后端实现影响，工程测试保留同分与绑定检查。

### 复核、范围与下一步

```powershell
Set-Location 'E:\秋招\项目相关\YOLO'
conda activate yolo
python -B -X utf8 scripts/check_backend_parity.py --check
```

该命令只读核对保存验收、固定输入、历史文件及代码 SHA，不重复执行模型、写文件或覆盖结果。首轮验收入口 `python -B -X utf8 scripts/check_backend_parity.py` 在报告已存在时拒绝覆盖。人工不需要再拍摄或审核。

**96 项工程测试通过；250 个历史文件 SHA / 修改时间保持一致。** 运行间交换的 90 份 raw 数组只放在本项目临时目录，任务结束后已清除；没有保留源图 / 标签副本。永久新增 4 份清单 / 代码 / 测试、一份集中 JSON 和一次 CUDA profile。摄像头仍使用 PyTorch；没有新训练、权重修改、ONNX 再导出、最终测试推理、质量 mAP 或正式 FPS。

M5-05 完成当时 **M5 5/6、主线 29/54，完整模块 3/9**。接续 M5-06 已完成固定 val5 的同条件部署质量对照，见下节。历史矩形输入下的 0.7807 不能直接当作部署降幅参照。

<a id="m5-06-validation-quality"></a>

## 10. M5-06：固定 val5 部署质量与可视化对照

### 先看结果

| 产物 | 内容 |
| --- | --- |
| [完整可视化页面](../demo/deployment/B1/M5-06/index.html) | 汇总指标图与 5 张四画面对照，离线打开即可 |
| [指标比较图](../demo/deployment/B1/M5-06/metrics_comparison.png) | PyTorch CUDA / ORT CPU / ORT CUDA 的总体、各类框和掩膜 mAP |
| [val01 三商品](../demo/deployment/B1/M5-06/val01_comparison.png) | 原图、人工标签、PyTorch、ONNX CUDA；三类同框，展示 3/3 |
| [val02 伊利](../demo/deployment/B1/M5-06/val02_comparison.png) | 单独伊利舒化，展示 1/1 |
| [val03 瑞幸漏检](../demo/deployment/B1/M5-06/val03_comparison.png) | 有 1 个标注杯，两个后端均未达到 0.25 门槛，展示 0/1 |
| [val04 三商品](../demo/deployment/B1/M5-06/val04_comparison.png) | 三类同框，展示 3/3 |
| [val05 无目标](../demo/deployment/B1/M5-06/val05_comparison.png) | 空背景 / 非目标物，0.25 门槛下无输出 |
| [各类指标 CSV](../demo/deployment/B1/M5-06/metrics_comparison.csv) | 三后端的各类 P / R、框与掩膜 AP，可用 Excel 打开 |
| [集中验收与诊断](../reports/deployment/M5-06_B1_validation_quality.json) | 实际推理、评价、逐图 / 逐实例配对、GPU / SHA 证据及低分边界问题 |

四画面上方是原图 / GT，下方是 PyTorch CUDA / ONNX CUDA；框与掩膜属于当前图。展示使用 conf=0.25、单标签规则，AP 使用 conf=0.001、多标签规则，没有将 200 多个低分评价候选全部画出来。

### 评价条件与实现

| 文件 | 职责 |
| --- | --- |
| [evaluate_deployment_B1.yaml](../configs/evaluate_deployment_B1.yaml) | 推理前冻结 val、640 方形 / FP32、两套门槛、GT 栅格化与 0.005 降幅上限 |
| [deployment_evaluation.py](../app/deployment_evaluation.py) | val5 防护、原始 YOLO 多边形读取、完整尺寸 GT、掩膜无损打包、官方指标核心适配与固定门槛统计 |
| [evaluate_deployment.py](../scripts/evaluate_deployment.py) | 三隔离推理进程、共同评价进程、可视化 / CSV、历史保护与只读复核 |
| [test_deployment_evaluation.py](../tests/test_deployment_evaluation.py) | 8 项工程测试：官方 GT 栅格化对照、非法标签、封存 / 配置防护、框成功掩膜失败、负样本误检、掩膜无损与降幅限值 |

仅使用冻结 val001～005，共 5 张 / 8 实例（山姆 2、伊利 3、瑞幸 3，含 1 张无目标图）。三个独立推理进程各执行 5 次 FP32 前向，同一共享 LetterBox 产生 batch1 / 640×640 / RGB tensor；评价候选为 conf=0.001、multi_label=True、class-aware NMS IoU=0.7、max_det=300，展示配置仍为 0.25 / 单标签。所有原始输出对照在既定容差内。

GT 来自冻结原 YOLO 多边形文本，乘原图尺寸后按锁定官方 `polygon2mask` 的 int32 截断 / fillPoly 方式生成每实例独立二值掩膜，downsample=1；框为多边形的浮点最小 / 最大坐标。预测框和 native 掩膜同样处于原图坐标。共同评价适配器实际调用锁定 `SegmentationValidator._process_batch`、官方 IoU 0.50～0.95 一对一匹配和 `SegmentMetrics` / `ap_per_class`。未调用官方整套训练验证加载器；这是有记录的原图掩膜评价条件。

历史训练评价使用 rect=True、降采样 / 重叠 GT 与默认验证掩膜路径。本次 0.7434 与历史 0.7807 的输入、GT / 掩膜处理条件不同，**不能把两数之差当作 ONNX 损失**。有效部署参照是本次同条件 PyTorch 的 0.7434。

### 同条件指标

| 后端 | 框 P / R | 框 mAP50 | 框 mAP50-95 | 掩膜 P / R | 掩膜 mAP50 | 掩膜 mAP50-95 | 掩膜部署降幅 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA | 0.9217 / 0.9746 | 0.9950 | 0.8205 | 0.9217 / 0.9746 | 0.9950 | 0.7434 | 参照 |
| ORT CPU | 0.9217 / 0.9746 | 0.9950 | 0.8205 | 0.9217 / 0.9746 | 0.9950 | 0.7434 | 0 |
| ORT CUDA | 0.9217 / 0.9746 | 0.9950 | 0.8205 | 0.9217 / 0.9746 | 0.9950 | 0.7434 | 0 |

两项 mAP50-95 均未测到降幅，通过预先固定的 0.005（0.5 个百分点）门槛。P / R 有约 1e-8 的插值数值差异；表中 P / R 是官方平滑平均 F1 最优操作点的均值，不是 conf=0.25 下的计数值，也不是“准确率”。

| 类别 | GT 实例 | 框 mAP50-95 | 掩膜 mAP50-95 | 0.25 下检出 / GT |
| --- | --- | --- | --- | --- |
| sam_whole_milk | 2 | 0.8950 | 0.5884 | 2/2 |
| yili_shuhua | 3 | 0.6970 | 0.7823 | 3/3 |
| luckin_cup | 3 | 0.8696 | 0.8595 | 2/3 |

三个后端的各类 AP 一致。固定 conf=0.25、IoU=0.5 时，框 / 掩膜均为 **TP=7、FP=0、FN=1，Precision=1.0、Recall=0.875**；val003 的瑞幸漏检明确保留。仅这 5 张验证图未出现误检，不能据此推断现场所有非目标都不会误检。

### 完整对齐与低分裁剪边界

展示门槛下的 10 组后端实例对照通过；0.001 门槛下，三个后端的独立后处理对官方同 raw 路径共 15 组比较均通过（对应 mask IoU 最低约 0.999947）。该低分路径存在少量差异像素，不能描述为逐像素全部相同。

低门槛跨后端有 **1/10 组未达到逐实例 mask IoU≥0.999**：ORT CPU 的 val001 候选 8324 在多标签规则下留下山姆 / 瑞幸两个分数约 0.00304 / 0.00244 的背景预测。PyTorch 的框 y1=686.0，CPU 为 686.0001221；按既有浮点框裁剪，CPU 删去 y=686 这一行的 417 个前景像素，因此两个相同候选掩膜的 IoU 为 0.96493。追加 2 次只读前向确认了坐标、差异行、面积及同类 GT mask IoU=0。

这两个预测没有进入 0.25 展示，对本批真实目标的 TP / AP 没有影响，最终两后端 mAP 一致。本次**按预设质量降幅门槛验收通过，同时保留低分实例对齐未通过记录**；没有通过改变裁剪、阈值或容差隐藏边界。B1 后处理仍保留原浮点几何规则。

ORT CPU / CUDA 推理进程均未导入 torch / ultralytics；实际 CUDA profile 记录 1325 次 GPU 节点事件、0 次 CPU 节点事件，来自正式 5 次 ORT CUDA 前向。原始数组与打包掩膜仅在临时目录交换，已全部清除；保存的是 5 张对照图、1 张指标图、CSV、HTML、1 份集中 JSON 和当次 profile。

### 复核与范围

```powershell
Set-Location 'E:\秋招\项目相关\YOLO'
conda activate yolo
python -B -X utf8 scripts/evaluate_deployment.py --check
```

只读核对已存配置 / 代码 / 资产 / 逐图用途及指标差异，不重复推理、写文件或覆盖可视化。首次评价入口为 `python -B -X utf8 scripts/evaluate_deployment.py`，已有输出拒绝覆盖。

**104 项工程测试通过；261 个历史文件 SHA / 修改时间不变。** 没有新照片、标签修改、训练、模型再导出、最终测试推理、摄像头切换或正式 FPS 结论。本批仍是同一实物包装、少量场景的验证，最终测试继续封存。

**M5 已完成 6/6，主线 30/54，完整模块 4/9。** 按完整路线下一项是 M6-01：核对商品 TensorRT 构建条件；若优先现场展示，也可先推进 M7-02 的 PyTorch / ONNX 摄像头后端选择，TensorRT 接入与 M7 完整验收仍依赖 M6。

## 11. M5 收尾文件清理

2026-10-04 按用户要求执行本阶段清理：**171 个可重建 Python 字节码文件、29 个缓存空目录已删除，原文件合计 2,298,510 字节（约 2.19 MiB）。** 对应 `.py` 源码全部保留，后续导入可重新生成缓存。

清理前后 1,754 个保留文件的 SHA256 / 修改时间一致，280 个部署相关文件哈希一致；`scripts/evaluate_deployment.py --check` 再次通过，连同 B1、M5-01～05 的保存证据一起核对。此次仅复核记录，不执行模型或重写验收结果。

ONNX / 元数据、集中参考数组、六份验收、四次 CUDA profile、M5-04 图片 / 视频、M5-06 五张对照与指标图 / CSV / HTML 均保留。CPU / CUDA 两张相同 PNG 是 M5-04 已登记的独立后端输出，仍被验收哈希引用；不能直接删除。M5 临时 raw / 打包掩膜原本已自动清除，本次没有新建空 ZIP，也没有处理此前留下的旧空目录树。

逐文件路径、缓存恢复来源和保留快照见[M5 清理清单](../reports/maintenance/20261004_M5_cleanup_manifest.json)，统一说明见[清理记录](../reports/maintenance/20261002_文件整理与清理清单.md#m5-cleanup)。工程状态仍为 M5 6/6、主线 30/54、完整模块 4/9。
