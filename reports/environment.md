# YOLO 项目环境报告

检查 / 更新日期：2026-10-02。当前完成任务：**M0-01 至 M0-04**，包含硬件、专用环境与源码锁定、正式检查及 E0 预训练演示。主线累计 4/54。

**M0-03 已完成：**训练环境通过；12:34 的正式复测在 DSHOW 下连续三次各读取三帧并正常释放，程序 exit code 0。之前的间歇性关闭超时已恢复，根因未确定。最新状态见[正式检查报告](environment_check.md)和第 11 节。

**M0-04 已完成：**官方预训练 YOLO11n-seg 已在 GPU 上运行图片、文件视频和摄像头；10 秒 / 297 帧演示正常保存并释放设备。三类映射、实际预测与漏检 / 误分类见[E0 报告](E0_预训练模型验证.md)及第 12 节。

本报告记录本机实际检查、环境创建、训练依赖安装和随机权重模型前向。前半部分保留 M0-01 的原始检查，当前安装状态见第 10 节。机器可读取证据见 [M0-01 检查快照](../logs/environment/M0-01_20261002_snapshot.json)、[M0-02 克隆快照](../logs/environment/M0-02_20261002_yolo_snapshot.json)与[安装验收快照](../logs/environment/M0-02_20261002_install_snapshot.json)。

## 1. 本次结论

已定位用户所说的 D 盘 Python 3.9 环境：

```text
环境名称：tf2
环境目录：D:\software\anaconda\envs\tf2
解释器：D:\software\anaconda\envs\tf2\python.exe
Python：3.9.23，64 位
```

该环境已有 `torch 2.8.0+cu129` 和 `torchvision 0.23.0+cu129`，实际 GPU 矩阵运算、卷积和 NMS 均通过。可以将它作为已验证的 Python/PyTorch 起点；完整 YOLO 训练、ONNX 和 TensorRT 链路尚未运行。

现已从 tf2 克隆创建项目专用环境 `D:\software\anaconda\envs\yolo`，Python、torch 和 torchvision 版本与上述起点一致；已安装训练依赖和项目内 Ultralytics 8.4.171 源码，随机权重 YOLO11n-seg 的 GPU 前向通过。后续项目工作使用 yolo，具体命令见[环境安装记录](../docs/环境安装记录.md)。

未激活环境的检查终端默认 `python` 指向 Anaconda base 的 Python 3.11.7，不会自动使用 yolo。后续命令需要激活正确环境或使用解释器绝对路径。

## 2. 硬件与操作系统

| 项目 | 实测信息 |
| --- | --- |
| 操作系统 | Windows 10 专业版，64 位，版本 10.0.19045 |
| CPU | Intel Core i5-14600KF |
| 系统内存 | 约 31.78 GiB 可报告物理内存，32 GB 级别 |
| 显卡 | NVIDIA GeForce RTX 5060 Ti |
| 显存 | 16311 MiB，约 16 GB 级别 |
| 驱动 | 576.88 |
| GPU compute capability | 12.0 |
| 显卡驱动模式 | WDDM |
| C / D / E 盘剩余空间 | 检查时约 38.9 / 46.4 / 301.2 GiB，随使用变化 |

硬件、基础运算及随机权重模型前向已验证，尚无训练 batch、训练时间或实时 FPS 的测量。项目数据、模型与结果继续按 E 盘项目路径组织。

## 3. 已有 Python 与 Conda 环境

Conda 管理程序位于 `D:\software\anaconda\Scripts\conda.exe`，版本 24.1.2。

| 环境 | 解释器 | Python | 检查到的 PyTorch / torchvision |
| --- | --- | --- | --- |
| Anaconda base，当前终端默认 | `D:\software\anaconda\python.exe` | 3.11.7 | 两者均未发现安装元数据 |
| tf2，用户所说的 3.9 环境 | `D:\software\anaconda\envs\tf2\python.exe` | 3.9.23 | 2.8.0+cu129 / 0.23.0+cu129 |
| yolo，新建项目专用环境 | `D:\software\anaconda\envs\yolo\python.exe` | 3.9.23 | 2.8.0+cu129 / 0.23.0+cu129，已验证 GPU 运算 |
| PEFT 项目环境 | `E:\秋招\项目相关\PEFT(Parameter-Efficient Fine-tuning)\.conda\python.exe` | 3.11.7 | 2.8.0+cu128 / 未发现 torchvision 元数据 |
| Intel Python | `E:\IntelSWTools2020\intelpython3\python.exe` | 3.7.7 | 本次仅检查解释器，未枚举依赖 |

补充完整依赖清点发现 tf2 及其克隆 yolo 均有 `tensorflow-gpu 2.10.0`；此前仅按发行包名 `tensorflow` 查询，未覆盖这一发行包名称。这里仅记录元数据，未验证 TensorFlow 运行。PEFT 和 Intel 环境仅做只读清点，不作为本项目当前运行环境。

### 明确调用项目 Python 3.9

激活方式：

```powershell
conda activate yolo
python --version
python -c "import sys; print(sys.executable)"
```

也可以直接调用已验证的解释器，避免依赖终端激活状态：

```powershell
& 'D:\software\anaconda\envs\yolo\python.exe' --version
& 'D:\software\anaconda\envs\yolo\python.exe' -m pip --version
```

激活后仍需检查解释器路径；环境选择应同时应用于编辑器、终端和训练脚本。

## 4. 三种 CUDA 信息

| 来源 | 当前值 | 含义 |
| --- | --- | --- |
| `nvidia-smi` 的 CUDA Version | 12.9 | 驱动报告的 CUDA 支持版本，不等于已安装 Toolkit |
| tf2 的 `torch.version.cuda` | 12.9 | 当前 PyTorch 构建使用的 CUDA 版本 |
| 当前 PATH 中的 `nvcc --version` | 11.2，V11.2.67 | 当前命令解析到的 CUDA Toolkit 编译器 |

当前 `nvcc` 位于 `D:\software\cuda\computing_development\bin\nvcc.exe`。本次没有全面清点机器所有位置的 Toolkit，因此这里记录的是当前可解析编译器。

尽管当前 Toolkit 编译器为 11.2，现有 PyTorch 的实际 CUDA 12.9 GPU 运算已成功。不能仅凭版本不同就判定 PyTorch 不可用，也不能据此保证后续原生扩展编译或 TensorRT 可用。驱动与运行时兼容性需结合软件版本和实际执行判断。[NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

后续涉及 CUDA 扩展编译、ORT CUDA DLL 或 TensorRT 时，应分别核查所选路线；本次没有更新驱动、Toolkit 或 PATH。

## 5. M0-01 初始 tf2 环境的关键依赖

| 包 | 实测版本 / 状态 |
| --- | --- |
| Python / pip | 3.9.23 / 25.2 |
| PyTorch | 2.8.0+cu129，已实际导入并执行 GPU 运算 |
| torchvision | 0.23.0+cu129，CPU/GPU NMS 已运行 |
| NumPy | 1.26.4 |
| pandas | 2.2.1 |
| matplotlib | 3.8.4 |
| SciPy | 1.13.1 |
| PyYAML | 6.0.3 |
| Ultralytics | 未发现包元数据与可导入模块 |
| OpenCV | 常见四种 opencv 发行包均未发现；`cv2` 模块未发现 |
| ONNX | 未发现包元数据与可导入模块 |
| ONNX Runtime CPU / GPU | 两种包均未发现；`onnxruntime` 模块未发现 |
| TensorRT Python bindings | 常见包元数据与 `tensorrt` 模块未发现 |

除 torch/torchvision 外，表中的已安装版本主要通过安装元数据读取，不代表逐包功能验证。TensorRT 这里只记录 Python 环境检查，没有全盘排查其他位置的原生 SDK。

Ultralytics 当前官方入门说明列出的基础要求是 Python ≥3.8、PyTorch ≥1.8；因此本次发现的 Python 3.9 不因该基础要求而必须更换。实际安装仍要核查锁定 Ultralytics 版本及 ONNX/ORT/TensorRT 的独立依赖要求。[官方安装说明](https://docs.ultralytics.com/quickstart/)

## 6. M0-01 初始 tf2 环境的 GPU 验证

测试使用 tf2 的解释器，运行结束后进程释放 GPU 资源。

| 检查 | 实际结果 |
| --- | --- |
| `torch.cuda.is_available()` | True |
| GPU 名称 | NVIDIA GeForce RTX 5060 Ti |
| GPU capability | (12, 0) |
| 当前 torch 构建架构列表 | 包含 sm_120 |
| `torch.backends.cudnn.version()` | 91002，原始返回值 |
| CUDA 矩阵乘法 | 16×16，输出有限，与 CPU 同输入结果一致 |
| CUDA Conv2d 前向 | 输入 1×3×64×64，输出 1×8×64×64，数值有限 |
| torchvision CPU NMS | 返回索引 [0, 2] |
| torchvision CUDA NMS | 返回索引 [0, 2]，与 CPU 一致 |

这是 M0-01 的小规模功能检查，不是模型精度、训练稳定性或性能基准。随后 yolo 环境中的随机权重项目模型前向见第 10 节，反向训练仍待开展。

## 7. 检查方法与复查命令

本次使用 `nvidia-smi`、PowerShell `Get-CimInstance`、Conda `env list --json`、各环境的解释器和 `importlib.metadata` 清点，再用 tf2 的 torch/torchvision 实际运行 GPU 检查。

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
& 'D:\software\anaconda\Scripts\conda.exe' env list --json
& 'D:\software\cuda\computing_development\bin\nvcc.exe' --version
& 'D:\software\anaconda\envs\tf2\python.exe' -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`py` 启动器未在当前 PATH 中找到，不影响显式调用已有解释器。

## 8. 状态边界与下一步

- **已完成：M0-01。** 硬件、驱动、已有 Python/Conda、当前 Toolkit 编译器及 Python 3.9 的基础 GPU 运算有记录。
- **已完成：M0-02。** 专用 yolo 环境、训练依赖、Ultralytics 可编辑源码、tag/commit 和路径规则均已验证，详见第 10 节。
- **已完成：M0-03。** 正式 `scripts/check_env.py` 已实现；18 项固定版本、源码、依赖一致性、OpenCV 和 GPU 通过，部署状态已登记；同一后端连续三次摄像头读取和释放通过，另有 10 秒采集记录。
- **已完成：M0-04。** 官方 E0 预训练权重、GPU 实际推理、三种输入与实例结果保存均通过，详见第 12 节。
- **未完成：M0-05、M0-06。** 未执行 ONNX 或 TensorRT 项目流程，也未进行完整环境恢复验收。
- **尚未通过或未检查：**摄像头长时间稳定性、项目训练 / 反向及正式精度评价、ONNX 导出及运行、TensorRT 构建及运行。随机权重与预训练模型推理、短时采集和正常关闭已证实；历史关闭超时的根因未确定。
- **环境变更：**M0-01 仅做检查；M0-02 从 tf2 离线克隆 yolo，初始 159 包版本一致；随后仅在 yolo 新增 6 包、升级 filelock，当前 165 包。既有 tf2 的关键版本复查未变，驱动、Toolkit 和全局 PATH 配置未调整。

下一步 M0-05 核查部署依赖，导出 ONNX 并实际执行，记录 TensorRT 构建 / 执行兼容性。后续摄像头优先使用本机已验证的 DSHOW。

## 9. yolo 专用环境创建与验收

环境为 `D:\software\anaconda\envs\yolo`，解释器为该目录下的 `python.exe`。Conda 离线克隆返回 exit code 0，并已在环境列表中登记。

对新解释器实际检查得到 Python 3.9.23、torch 2.8.0+cu129、torchvision 0.23.0+cu129、PyTorch CUDA 12.9；torch/torchvision 的加载位置均在新环境中。矩阵乘法、Conv2d 和 CUDA NMS 通过。源环境与新环境的 159 个发行包版本一致。

创建命令和使用方法见[环境安装记录](../docs/环境安装记录.md)，过程见[Conda 创建日志](../logs/environment/M0-02_20261002_conda_clone.log)。本节保留克隆时状态；随后安装与完整 M0-02 验收如下。

## 10. 源码与训练安装验收

| 项目 | 当前实际结果 |
| --- | --- |
| 解释器 | `D:\software\anaconda\envs\yolo\python.exe`，Python 3.9.23 |
| torch / torchvision | 2.8.0+cu129 / 0.23.0+cu129，保留已有 GPU 构建 |
| Ultralytics | 8.4.171，tag `v8.4.171` |
| 实际 commit | `86f5c8c401a630da651d321e691ab0af540812cf` |
| 导入方式 | 项目 `third_party/ultralytics/ultralytics/__init__.py`；安装元数据 editable=True |
| OpenCV / NumPy | 4.11.0.86 / 1.26.4，PNG 内存编解码通过 |
| 完整依赖检查 | `pip check` 无冲突；发行包总数 165 |
| 新增发行包 | Ultralytics、OpenCV、polars、polars-runtime-32、nvidia-ml-py、ultralytics-thop，共 6 包 |
| 版本变化 | filelock 3.13.1 → 3.19.1；没有移除其他发行包 |
| 部署包 | ONNX、ORT、TensorRT 仍未安装；M0-05 单独解析和验证 |

训练依赖见 [requirements-train.txt](../requirements-train.txt)，约束见 [configs/constraints-train.txt](../configs/constraints-train.txt)，源码锁定见 [configs/source-lock.json](../configs/source-lock.json)。源码恢复脚本只在 commit 一致时继续；上游目录当前工作树无源码修改。

运行 [scripts/verify_training_setup.py](../scripts/verify_training_setup.py) 返回 exit code 0。由官方 YAML 构建随机初始化的 YOLO11n-seg，n 规模、80 类、Segment 头，参数 2876848；FP32 输入 `[1,3,640,640]` 实际在 RTX 5060 Ti / cuda:0 上前向，全部输出有限。推理候选为 `[1,116,8400]`，原型为 `[1,32,160,160]`，其余原始输出结构见快照。该 M0-02 结果仅说明随机权重模型可运行；之后 E0 的实际推理见第 12 节，项目训练仍未开展。

已建立必要一级目录、`artifacts/pretrained/`、路径配置及统一解析函数。从项目外工作目录调用 `deploy.paths.load_paths()` 也能得到正确项目路径。项目本机 Ultralytics 配置位于忽略的 `logs/ultralytics_settings/`，使用项目 data、pretrained 和 runs 路径。

安装时官方 PyPI 大文件下载停滞；改用清华镜像后完成，下载的 6 个发行包哈希与官方 PyPI 元数据匹配，实际安装档案也已复核。原 tf2 的 torch、torchvision、NumPy、protobuf、filelock 版本复查未变，且仍未安装 Ultralytics。

本次证据：[安装日志](../logs/environment/M0-02_20261002_pip_install.log)、[下载哈希](../logs/environment/M0-02_20261002_download_hashes.json)、[包变化](../logs/environment/M0-02_20261002_package_changes.json)、[安装验收快照](../logs/environment/M0-02_20261002_install_snapshot.json)、[验收日志](../logs/environment/M0-02_20261002_verify.log)。完整恢复和重启后的 E0/ONNX 验收继续保留在 M0-06。

## 11. M0-03 正式检查入口与摄像头复测

已实现 [scripts/check_env.py](../scripts/check_env.py)，复用 M0-02 的检查函数，并增加 requirements 中 18 项直接依赖的固定版本比对、部署模块状态和限时摄像头子进程。针对间歇性问题，摄像头默认要求同一后端连续三次打开、各读取三帧、正常释放，记录各阶段耗时。任一次失败都不能通过该后端。每次保存独立 JSON 快照，生成最近一次的 [environment_check.md](environment_check.md)；人工维护的本报告继续保留。

最后一次正式检查：[2026-10-02 12:34:01 快照](../logs/environment/M0-03_20261002_123401_124014_snapshot.json)、[执行日志](../logs/environment/M0-03_20261002_recheck.log)。从项目外工作目录调用，训练检查与摄像头检查均通过，程序 exit code 0。DSHOW 三次的释放耗时均为 0.273 秒；12:32 的前一轮正式检查也通过。

此前 [11:44:08 失败快照](../logs/environment/M0-03_20261002_114408_074569_snapshot.json)和[失败执行日志](../logs/environment/M0-03_20261002_check_final.log)继续保留。本轮开始复测时仍重现过一次“读取成功、释放超时”，随后原始方式恢复正常；不能把后来通过归因于某项未经验证的修改。

| 摄像头检查 | 实际发现 |
| --- | --- |
| Windows 设备 | FHD USB camera；PnP 问题码 0；驱动提供者 Microsoft，版本 10.0.19041.6033 |
| 隐私权限清点 | HKCU / HKLM webcam ConsentStore 为 Allow，不代表已经排除其他软件占用 |
| DSHOW / 索引 0，当前 | 两轮正式检查均连续三次读取和释放通过；`[480,640,3]` / uint8，像素最大值 255 |
| DSHOW 短时持续采集 | 10.003 秒读取 286 帧，读取失败 0 次，随后 release 耗时 0.274 秒；不是 YOLO 应用 FPS 或长时间稳定性验收 |
| 历史 DSHOW 异常 | 读取三帧成功，但 release 超时；本轮最小程序还出现过释放约 7.6 秒，之后正常 |
| 历史 MSMF 异常 | 初始化成功但读帧失败，`can't grab frame` / `-2147483638`；恢复后的正式检查使用 DSHOW，没有重新验收 MSMF |
| 历史参数 / 版本对照 | 增加等待、MJPG、关闭 MSMF 硬件转换、临时 OpenCV 4.10 均未解除当时的问题 |
| 当前环境 | 仍为 opencv-python 4.11.0.86，没有更换驱动、调整 USB 配置或采用 COM 修改 |
| ORT / TensorRT 状态 | 模块未安装；provider 列表未获得，实际部署执行待 M0-05 |

用户已确认其他摄像头程序都已关闭。原始方式与 COM MTA 方式的交替对照均通过，不能证明 COM 是根因。当前已确认 DSHOW 恢复可用，尚未确定历史异常的根因；没有证据将它归因于 YOLO 依赖或硬件损坏，也不能宣称驱动问题已被永久修复。M0-03 验收时主线为 3/54；之后 M0-04 完成后为 4/54；M7 保留持续运行验收。

本轮补充证据：[原检查连续复开记录](../logs/environment/M0-03_20261002_recheck_worker.json)、[最小程序及采集时长对照](../logs/environment/M0-03_20261002_recheck_variants.json)、[COM 交替对照](../logs/environment/M0-03_20261002_recheck_paired_com.json)、[10 秒采集记录](../logs/environment/M0-03_20261002_recheck_stream.json)。

操作说明、参数、退出码与诊断证据见 [docs/环境检查使用说明.md](../docs/环境检查使用说明.md)。M0-03 检查本身没有保存或显示摄像头画面；M0-04 的演示产出另见下节。设备报告帧率未作为应用 FPS。

## 12. M0-04：E0 预训练模型实际运行

已下载并固定官方 `yolo11n-seg.pt`，新增 [scripts/prepare_e0.py](../scripts/prepare_e0.py)、[scripts/predict_e0.py](../scripts/predict_e0.py) 和 [configs/e0.yaml](../configs/e0.yaml)。保留原模型 80 类，通过实际 `model.names` 得到 cup=41、bottle=39、cell phone=67，另外记录项目三类 ID 0、1、2。

| 输入 | 实际验证 |
| --- | --- |
| 官方 bus.jpg | GPU FP32 推理得到 4 个 person、1 个 bus 及 5 张非空 mask |
| 自采桌面图 | 三类筛选预测到 cup、cell phone，保存原图 / 结果图；从项目外工作目录运行成功 |
| 摄像头 | DSHOW 10.003 秒处理 297 帧，保存原始 / 结果 MP4；正常释放耗时 0.270 秒 |
| 同段文件视频 | 独立进程完整处理 297 帧，保存可视化；从项目外工作目录运行成功，文件资源正常释放 |
| 输出核对 | 原始 / 结果视频可重新完整解码，帧数一致；框与 mask 数匹配、数值有限、mask 非空且输出在 cuda:0 |

用户确认右侧杯子后方黑色矩形为手机、最右物体为瓶子，另有左侧被手遮挡的手机。右侧手机有预测；左侧遮挡手机漏检；瓶子没有 bottle 输出，部分帧被预测为 cup。这些逐例现象不是数据集精度指标，也没有证据将它们单独归因于某个原因。

本次没有更换依赖、驱动或锁定源码。脚本、参数、权重来源、输入和预测证据见 [E0 使用说明](../docs/E0_预训练模型使用说明.md)、[E0 报告](E0_预训练模型验证.md)、[产出验收 JSON](../logs/environment/M0-04_20261002_artifact_validation.json)。下一步 M0-05；项目训练、独立部署和持续运行稳定性继续按模块验收。
