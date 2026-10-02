"""Run M0-03 checks and save a JSON snapshot plus a readable environment report."""

import argparse
from datetime import datetime
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def camera_worker(index, backend, mode):
    """Read three frames and release the device; do not save or display images."""
    import cv2

    started = time.perf_counter()
    api = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}[backend]
    print("stage=opening", file=sys.stderr, flush=True)
    capture = cv2.VideoCapture(index, api)
    result = {"index": index, "requested_backend": backend, "mode": mode,
              "passed": False, "valid_frames": 0, "released": False,
              "opencv_version": cv2.__version__, "opencv_file": cv2.__file__,
              "open_seconds": round(time.perf_counter() - started, 3)}
    try:
        result["opened"] = capture.isOpened()
        print(f"stage=opened success={result['opened']}", file=sys.stderr, flush=True)
        if not result["opened"]:
            result["error"] = "Camera could not be opened"
            return result
        result["actual_backend"] = capture.getBackendName()
        if mode == "mjpeg640":
            result["format_requests"] = {
                "MJPG": capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                "width_640": capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
                "height_480": capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            }
        reading_started = time.perf_counter()
        for attempt in range(10):
            print(f"stage=reading attempt={attempt + 1}", file=sys.stderr, flush=True)
            read_ok, frame = capture.read()
            if read_ok and frame is not None and frame.size and frame.ndim == 3:
                result["valid_frames"] += 1
                result["frame_shape"] = list(frame.shape)
                result["frame_dtype"] = str(frame.dtype)
                result["frame_min"] = int(frame.min())
                result["frame_max"] = int(frame.max())
                if result["valid_frames"] == 3:
                    result["passed"] = True
                    break
        result["read_seconds"] = round(time.perf_counter() - reading_started, 3)
        result["reported_fps"] = capture.get(cv2.CAP_PROP_FPS)
        if not result["passed"]:
            result["error"] = "Could not read three valid frames in ten attempts"
        return result
    finally:
        print("camera_result=" + json.dumps(result), file=sys.stderr, flush=True)
        print("stage=releasing", file=sys.stderr, flush=True)
        releasing_started = time.perf_counter()
        capture.release()
        result["released"] = True
        result["release_seconds"] = round(time.perf_counter() - releasing_started, 3)
        print("stage=released", file=sys.stderr, flush=True)


def check_camera(index, timeout, backend, mode, cycles=3):
    """Require consecutive read/release cycles on one backend, using bounded workers."""
    attempts = []
    completed_cycles = 0
    backends = (["DSHOW", "MSMF"] if platform.system() == "Windows" else ["ANY"]) if backend == "AUTO" else [backend]
    for backend in backends:
        command = [sys.executable, str(Path(__file__).resolve()), "--camera-worker",
                   "--camera-index", str(index), "--camera-backend", backend, "--camera-mode", mode]
        for cycle in range(1, cycles + 1):
            started = time.perf_counter()
            try:
                worker = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=timeout,
                                        env={**os.environ, "PYTHONUTF8": "1"})
                if worker.returncode == 0:
                    attempt = json.loads(worker.stdout)
                    if not attempt.get("released"):
                        attempt["passed"] = False
                        attempt["error"] = "Worker did not confirm device release"
                else:
                    attempt = {"requested_backend": backend, "passed": False,
                               "error": worker.stderr.strip() or worker.stdout.strip()}
                if worker.stderr.strip():
                    attempt["driver_messages"] = worker.stderr.strip()
            except subprocess.TimeoutExpired as error:
                attempt = {"requested_backend": backend, "passed": False, "error": f"Timeout after {timeout}s"}
                if error.stderr:
                    attempt["driver_messages"] = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else error.stderr
                    for line in attempt["driver_messages"].splitlines():
                        if line.startswith("camera_result="):
                            partial = json.loads(line.split("=", 1)[1])
                            partial["frames_read"] = partial.pop("passed")
                            attempt.update(partial)
            except (OSError, ValueError) as error:
                attempt = {"requested_backend": backend, "passed": False, "error": str(error)}
            attempt["cycle"] = cycle
            attempt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            attempts.append(attempt)
            if not attempt["passed"]:
                break
            completed_cycles = max(completed_cycles, cycle)
            if cycle == cycles:
                return {"status": "passed", "index": index, "mode": mode,
                        "required_cycles": cycles, "completed_cycles": cycles,
                        "selected_backend": backend, "attempts": attempts}
    return {"status": "failed", "index": index, "mode": mode,
            "required_cycles": cycles, "completed_cycles": completed_cycles,
            "selected_backend": None, "attempts": attempts}


def check_deployment():
    """Record installed deployment modules and ORT provider availability only."""
    checks = {}
    for name in ("onnx", "onnxruntime", "tensorrt"):
        if importlib.util.find_spec(name) is None:
            checks[name] = {"status": "not_installed", "next_task": "M0-05"}
            continue
        try:
            module = __import__(name)
            checks[name] = {"status": "imported", "version": module.__version__, "loaded_file": module.__file__}
            if name == "onnxruntime":
                checks[name]["available_providers"] = module.get_available_providers()
                checks[name]["model_execution_checked"] = False
        except Exception as error:
            checks[name] = {"status": "import_failed", "error": f"{type(error).__name__}: {error}", "next_task": "M0-05"}
    return checks


def write_markdown(report, snapshot):
    """Write the latest machine-check report; preserve the curated environment.md."""
    output = ROOT / "reports/environment_check.md"
    relative_snapshot = os.path.relpath(snapshot, output.parent).replace("\\", "/")
    training = report["training"]
    camera = report["camera"]
    lines = ["# M0-03 环境检查报告", "", f"检查时间：{report['checked_at']}。",
             f"任务验收：**{'通过' if report['task_complete'] else '未通过或未完成'}**。",
             f"原始证据：[本次 JSON 快照]({relative_snapshot})。", "",
             "## 1. 解释器与训练环境", "", "| 检查 | 结果 |", "| --- | --- |",
             f"| Python | {report['python']['version']} |",
             f"| 解释器 | `{report['python']['executable']}` |",
             f"| 训练环境检查 | {training['status']} |"]
    if training["status"] == "passed":
        details = training["details"]
        lines += [f"| Ultralytics | {details['source']['version']}，editable=True |",
                  f"| 源码 commit | `{details['source']['actual_commit']}` |",
                  f"| GPU | {details['gpu']['name']} |",
                  f"| torch / CUDA | {details['gpu']['torch']} / {details['gpu']['cuda']} |",
                  f"| 依赖检查 | {details['pip_check']} |",
                  "| 模型运行 | 随机权重 YOLO11n-seg，640×640，FP32，GPU 输出有限 |",
                  "| OpenCV | PNG 编解码后像素一致 |"]
    else:
        lines += ["", "```text", training["error"], "```"]
    lines += ["", "## 2. 摄像头", "", f"状态：**{camera['status']}**；设备索引：{camera['index']}。", ""]
    for attempt in camera.get("attempts", []):
        lines += [f"- {attempt['requested_backend']} / 第 {attempt['cycle']} 次：{'通过' if attempt['passed'] else '失败'}。"]
        if attempt["passed"]:
            lines += [f"  已读取 {attempt['valid_frames']} 帧，形状 {attempt['frame_shape']}，"
                      f"类型 {attempt['frame_dtype']}，设备报告帧率 {attempt['reported_fps']}。"]
            lines += [f"  设备正常释放：{attempt['released']}；打开 / 读取 / 释放耗时："
                      f"{attempt['open_seconds']} / {attempt['read_seconds']} / {attempt['release_seconds']} 秒。"]
        else:
            lines += [f"  错误：{attempt['error']}"]
            if attempt.get("valid_frames"):
                lines += [f"  超时前读到 {attempt['valid_frames']} 帧，形状 {attempt['frame_shape']}；"
                          f"设备释放完成：{attempt['released']}。"]
            if attempt.get("driver_messages"):
                lines += ["", "```text", attempt["driver_messages"].strip(), "```", ""]
    lines += ["", "检查尝试读取少量帧并释放设备；驱动卡住时终止该检查的子进程。没有保存或显示画面。设备报告帧率不是实测应用 FPS。",
              "", "## 3. 部署环境状态", "", "| 模块 | 当前状态 | 后续 |", "| --- | --- | --- |"]
    for name, result in report["deployment"].items():
        details = result["status"]
        if "available_providers" in result:
            details += "；providers=" + ", ".join(result["available_providers"])
        if "error" in result:
            details += "；" + result["error"]
        lines += [f"| {name} | {details} | M0-05 安装/执行验证 |"]
    lines += ["", "部署依赖缺失在本阶段如实登记，不作为训练环境或摄像头检查已通过的证据。provider 列表也不等同于模型实际执行。",
              "", "## 4. 验收边界与下一步", "",
              f"本次摄像头要求在同一后端连续完成 {camera.get('required_cycles', 3)} 次打开、读取三帧和正常释放；任一次失败均不能通过该后端。",
              "M0-03 验收要求：训练检查通过、摄像头重复读取和释放通过，并记录部署模块状态。",
              "尚未验证：E0 预训练模型、训练/反向、ONNX 模型执行、TensorRT 构建/执行和长时间摄像头稳定性。"]
    lines += ["下一步 M0-04：运行预训练 YOLO11n-seg，检查图片、视频和摄像头中的目标框与实例掩膜。"
              if report["task_complete"] else "下一步：解决报告中的未通过项并重新检查，再验收 M0-03。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-timeout", type=float, default=15.0, help="Seconds allowed per camera cycle")
    parser.add_argument("--camera-cycles", type=int, default=3, help="Consecutive successful cycles required on one backend")
    parser.add_argument("--skip-camera", action="store_true", help="Record camera as skipped; M0-03 stays incomplete")
    parser.add_argument("--output", help="JSON snapshot path, relative to the project root or absolute")
    parser.add_argument("--camera-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--camera-backend", choices=["AUTO", "DSHOW", "MSMF", "ANY"], default="AUTO")
    parser.add_argument("--camera-mode", choices=["native", "mjpeg640"], default="native",
                        help="native preserves defaults; mjpeg640 requests MJPG at 640x480")
    args = parser.parse_args()
    if args.camera_index < 0 or args.camera_timeout <= 0 or args.camera_cycles < 1:
        parser.error("camera-index must be nonnegative; camera-timeout and camera-cycles must be positive")
    if args.camera_worker:
        print(json.dumps(camera_worker(args.camera_index, args.camera_backend, args.camera_mode)))
        return 0

    checked_at = datetime.now().astimezone()
    report = {"task": "M0-03", "checked_at": checked_at.isoformat(),
              "project_root": str(ROOT), "cwd": str(Path.cwd()),
              "python": {"version": platform.python_version(), "executable": sys.executable, "prefix": sys.prefix},
              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}}
    print("Checking training environment, source and GPU...", flush=True)
    try:
        from scripts.verify_training_setup import check_training_setup

        details = check_training_setup()
        for key in ("task", "date", "not_checked"):
            details.pop(key, None)
        report["training"] = {"status": "passed", "details": details}
    except Exception as error:
        report["training"] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc()}
    report["deployment"] = check_deployment()
    if args.skip_camera:
        report["camera"] = {"status": "skipped", "index": args.camera_index}
    else:
        print(f"Checking camera index {args.camera_index}...", flush=True)
        report["camera"] = check_camera(args.camera_index, args.camera_timeout, args.camera_backend,
                                        args.camera_mode, args.camera_cycles)
    report["task_complete"] = report["training"]["status"] == "passed" and report["camera"]["status"] == "passed"
    stamp = checked_at.strftime("%Y%m%d_%H%M%S_%f")
    snapshot = Path(args.output) if args.output else Path(f"logs/environment/M0-03_{stamp}_snapshot.json")
    snapshot = (snapshot if snapshot.is_absolute() else ROOT / snapshot).resolve()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readable = write_markdown(report, snapshot)
    print(json.dumps({"task_complete": report["task_complete"], "training": report["training"]["status"],
                      "camera": report["camera"], "deployment": report["deployment"],
                      "snapshot": str(snapshot), "report": str(readable)}, ensure_ascii=False, indent=2))
    if report["task_complete"]:
        return 0
    return 2 if report["camera"]["status"] == "skipped" and report["training"]["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
