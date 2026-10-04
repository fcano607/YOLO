# M5：商品 ONNX 独立部署与接口说明

更新：2026-10-04。**M5-01 已完成，M5 1/6，主线 25/54，完整模块仍 3/9。** 已从冻结 B1 / E1-A best 导出商品 ONNX，检查图结构、类别与接口，并完成三张已有 val 图的 ORT CPU / CUDA 原始输出检查。共享预处理、独立 NMS / 掩膜还原、完整图片 / 视频程序、30 输入配对和部署质量评价仍待 M5-02～06，最终 test5 继续封存。

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

## 3. 前后处理接口约定（实现待后续任务）

**M5-02 预处理约定：**原图 BGR / uint8 / HWC → 保持比例缩放 → 居中填充到 640×640，填充值 114 → RGB → float32 / 255 → NCHW / 连续内存。首版 `auto=false`、`scale_fill=false`、`scaleup=true`，使用 OpenCV 双线性缩放。必须保留原图尺寸、缩放比例和上下左右整数补边，供框与 mask 还原。

锁定 LetterBox 的比例为 `r=min(640/h,640/w)`，缩放宽高分别为 `round(w*r)`、`round(h*r)`；单边补量采用 `round(half_pad-0.1)` 与 `round(half_pad+0.1)`，处理奇数补边。M5-01 的参考数组采用官方 LetterBox 生成，只用于导出对照，不能当成已经完成了自写预处理。

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

**下一步 M5-02：**实现共享 NumPy/OpenCV 预处理，在横图、竖图与奇数尺寸输入上核对颜色、像素值和补边，再将这一接口同时供 PyTorch 与 ORT 使用。完成后可确保两种后端接收完全相同的张量，随后开展 M5-03 的独立框 / 掩膜后处理。无需新增人工标注。
