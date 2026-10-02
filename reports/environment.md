# YOLO 项目环境报告

检查日期：2026-10-02。当前完成任务：**M0-01，确认硬件和已有环境**。

本报告记录本机实际检查结果；项目依赖安装、源码锁定、摄像头和模型验证在后续任务完成后补充。机器可读取证据见 [M0-01 检查快照](../logs/environment/M0-01_20261002_snapshot.json)。

## 1. 本次结论

已定位用户所说的 D 盘 Python 3.9 环境：

```text
环境名称：tf2
环境目录：D:\software\anaconda\envs\tf2
解释器：D:\software\anaconda\envs\tf2\python.exe
Python：3.9.23，64 位
```

该环境已有 `torch 2.8.0+cu129` 和 `torchvision 0.23.0+cu129`，实际 GPU 矩阵运算、卷积和 NMS 均通过。可以将它作为已验证的 Python/PyTorch 起点；完整 YOLO 训练、ONNX 和 TensorRT 链路尚未运行。

当前检查终端默认 `python` 指向 Anaconda base 的 Python 3.11.7，不会自动使用 tf2。后续命令需要激活正确环境或使用解释器绝对路径。

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

当前仅能确认硬件与基础运算，尚无项目 batch、训练时间或实时 FPS 的测量。项目数据、模型与结果继续按 E 盘项目路径组织。

## 3. 已有 Python 与 Conda 环境

Conda 管理程序位于 `D:\software\anaconda\Scripts\conda.exe`，版本 24.1.2。

| 环境 | 解释器 | Python | 检查到的 PyTorch / torchvision |
| --- | --- | --- | --- |
| Anaconda base，当前终端默认 | `D:\software\anaconda\python.exe` | 3.11.7 | 两者均未发现安装元数据 |
| tf2，用户所说的 3.9 环境 | `D:\software\anaconda\envs\tf2\python.exe` | 3.9.23 | 2.8.0+cu129 / 0.23.0+cu129 |
| PEFT 项目环境 | `E:\秋招\项目相关\PEFT(Parameter-Efficient Fine-tuning)\.conda\python.exe` | 3.11.7 | 2.8.0+cu128 / 未发现 torchvision 元数据 |
| Intel Python | `E:\IntelSWTools2020\intelpython3\python.exe` | 3.7.7 | 本次仅检查解释器，未枚举依赖 |

环境名称 tf2 不能证明安装了 TensorFlow；本次该环境未发现 TensorFlow 包元数据。PEFT 和 Intel 环境仅做只读清点，不作为本项目当前运行环境。

### 明确调用 Python 3.9

激活方式：

```powershell
conda activate tf2
python --version
python -c "import sys; print(sys.executable)"
```

也可以直接调用已验证的解释器，避免依赖终端激活状态：

```powershell
& 'D:\software\anaconda\envs\tf2\python.exe' --version
& 'D:\software\anaconda\envs\tf2\python.exe' -m pip --version
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

## 5. Python 3.9 环境的关键依赖

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

## 6. 实际 GPU 验证

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

这是小规模功能检查，不是 YOLO 模型精度、训练稳定性或性能基准。尚未运行反向训练和项目模型。

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
- **未完成：M0-02 至 M0-06。** 尚未建立并锁定 YOLO 项目环境/源码，未编写正式 `scripts/check_env.py`，未执行 E0、ONNX 或 TensorRT 项目流程。
- **尚未检查：**摄像头读取、YOLO 前向与训练、ONNX 导出及运行、TensorRT 构建及运行。
- **本次环境变更：**未安装、卸载或升级任何依赖，未修改现有环境、驱动、Toolkit 和 PATH 配置。

下一任务为 M0-02：沿用已经验证的 Python 3.9 / PyTorch 组合作为候选起点，确定独立项目环境方案及 Ultralytics 源码版本，核查后续部署依赖，再执行安装与版本记录。保留用户原有 tf2 工作环境，避免直接在其中叠加未经验证的项目依赖；新环境名称和路径在实际建立后更新，本报告不将其写为已存在。
