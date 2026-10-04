"""M5-06: isolated val5 inference, a common pinned metric core and visual comparisons."""
import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from app.deployment_evaluation import (CONFIG, configuration, evaluate_instances, ground_truth,
    metric_deltas, pack_instances, unpack_instances)
from app.product_data import image_read, image_write, write_json
from deploy.base_backend import create_backend, file_sha256, load_bundle, read_json
from deploy.onnx_contract import compare_raw
from deploy.parity import TOLERANCE, assert_binding, compare_instances
from deploy.postprocess import postprocess_b1
from deploy.preprocess import preprocess_bgr
from scripts.check_backend_parity import child, check_hashes, official_instances, ref, tensor_sha

REPORT = ROOT / "reports/deployment/M5-06_B1_validation_quality.json"
PREVIOUS = ROOT / "reports/deployment/M5-05_B1_backend_parity.json"
DEMO = ROOT / "demo/deployment/B1/M5-06"
ROUTES = ("torch_cuda", "ort_cpu", "ort_cuda")
IMPLEMENTATION = ("configs/evaluate_deployment_B1.yaml", "app/deployment_evaluation.py",
                  "scripts/evaluate_deployment.py", "tests/test_deployment_evaluation.py")
METRIC_SOURCE = tuple("third_party/ultralytics/ultralytics/" + p for p in (
    "models/yolo/segment/val.py", "models/yolo/detect/val.py", "engine/validator.py",
    "utils/metrics.py", "data/utils.py"))
COLORS = ((40, 140, 255), (255, 160, 40), (190, 70, 180))
LABELS = ("Sam milk", "Yili milk", "Luckin cup")


def check_image(sample):
    if sample["split"] != "val" or file_sha256(ROOT / sample["image_path"]) != sample["image_sha256"]:
        raise ValueError("Only unchanged frozen validation images are allowed")
    return image_read(ROOT / sample["image_path"])


def inference_worker(route, scratch, profile_prefix=None):
    policy, samples = configuration()
    bundle = load_bundle()
    backend = create_backend(bundle, "torch" if route == "torch_cuda" else "onnx",
                             "cpu" if route == "ort_cpu" else "cuda:0", profile_prefix)
    cases = []
    try:
        identity = backend.describe()
        for index, sample in enumerate(samples, 1):
            image = check_image(sample)
            images, geometry = preprocess_bgr(image)
            raw = backend.run_raw(images)
            metric = postprocess_b1(*raw, geometry, **policy["metric_postprocess"])
            display = postprocess_b1(*raw, geometry, **policy["display_postprocess"])
            assert_binding(metric, raw[0]); assert_binding(display, raw[0])
            case = {"image_id": sample["image_id"], "input_sha256": tensor_sha(images),
                    "metric_count": len(metric.scores), "display_count": len(display.scores),
                    "metric_candidate_cap_reached": metric.counts["after_nms"] >= 300}
            if route != "torch_cuda":
                with np.load(scratch / ("%02d_torch_cuda.npz" % index), allow_pickle=False) as reference:
                    if str(reference["input_sha256"]) != tensor_sha(images):
                        raise ValueError("Input tensors differ between backends")
                    case["raw_comparison"] = compare_raw((reference["output0"], reference["output1"]), raw,
                                                         TOLERANCE["raw"])
                    if not all(v["allclose"] for v in case["raw_comparison"]):
                        raise ValueError("Raw backend parity exceeds the fixed tolerance")
            np.savez(scratch / ("%02d_%s.npz" % (index, route)), output0=raw[0], output1=raw[1],
                     input_sha256=np.array(tensor_sha(images)), **pack_instances(metric, "metric_"),
                     **pack_instances(display, "display_"))
            cases.append(case)
    finally:
        backend.close()
    result = {"route": route, "backend": identity, "cases": cases, "backend_closed": backend.closed,
              "torch_imported": "torch" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules,
              "profile": ref(backend.profile_path) if getattr(backend, "profile_path", None) else None}
    if route.startswith("ort") and (result["torch_imported"] or result["ultralytics_imported"]):
        raise ValueError("Independent ORT inference imported a training framework")
    print("__QUALITY_INFERENCE__=" + json.dumps(result, ensure_ascii=False))


def evaluation_worker(scratch):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    policy, samples = configuration()
    names = load_bundle().class_names
    results, official_checks, parity_checks = {}, [], []
    for route in ROUTES:
        def records():
            for index, sample in enumerate(samples, 1):
                image = check_image(sample)
                _, geometry = preprocess_bgr(image)
                truth = ground_truth(sample, image.shape[:2])
                with np.load(scratch / ("%02d_%s.npz" % (index, route)), allow_pickle=False) as saved:
                    metric, display = unpack_instances(saved, "metric_"), unpack_instances(saved, "display_")
                    raw, proto = saved["output0"], saved["output1"]
                official = official_instances(raw, proto, geometry, image, policy["metric_postprocess"])
                comparison = compare_instances(metric, official)
                if not comparison["passed"]:
                    raise ValueError("Independent metric postprocessing differs from the official same-raw path")
                official_checks.append({"route": route, "image_id": sample["image_id"], "comparison": comparison})
                del official, raw, proto
                if route != "torch_cuda":
                    with np.load(scratch / ("%02d_torch_cuda.npz" % index), allow_pickle=False) as saved:
                        reference = unpack_instances(saved, "metric_")
                        parity = compare_instances(metric, reference)
                        del reference
                        display_parity = compare_instances(display, unpack_instances(saved, "display_"))
                    if not display_parity["passed"]:
                        raise ValueError("Display-threshold backend parity failed")
                    parity_checks.append({"route": route, "image_id": sample["image_id"],
                                          "metric_threshold_comparison": parity, "display_comparison": display_parity})
                yield sample["image_id"], truth, metric, display
        results[route] = evaluate_instances(records(), names)
    deltas = {r: metric_deltas(results["torch_cuda"]["metrics"], results[r]["metrics"], policy["max_map50_95_drop"])
              for r in ("ort_cpu", "ort_cuda")}
    if not all(v["passed"] for v in deltas.values()):
        raise ValueError("Deployment mAP50-95 drop exceeds the predefined 0.005 limit")
    print("__QUALITY_METRICS__=" + json.dumps({"routes": results, "deltas": deltas,
        "official_metric_postprocessing": official_checks, "backend_instance_parity": parity_checks,
        "model_inference": False, "metric_device": "cpu"}, ensure_ascii=False))


def panel(image, instances, title, scores=True):
    canvas = image.copy()
    if instances is not None:
        for mask, cls in zip(instances.masks, instances.class_ids):
            selected = mask.astype(bool)
            canvas[selected] = np.rint(.65*canvas[selected] + .35*np.array(COLORS[int(cls)])).astype(np.uint8)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (640, 360), interpolation=cv2.INTER_AREA)
    if instances is not None:
        for box, score, cls in zip(instances.boxes_xyxy, instances.scores, instances.class_ids):
            x1, y1, x2, y2 = np.rint(box*np.array([640/w,360/h,640/w,360/h])).astype(int)
            color = COLORS[int(cls)]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
            text = LABELS[int(cls)] + (" %.3f" % score if scores else " GT")
            cv2.putText(canvas, text, (max(0,x1),max(18,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    banner = np.full((44,640,3), 245, np.uint8)
    cv2.putText(banner,title,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.68,(30,30,30),1)
    return np.vstack((banner,canvas))


def render_visuals(scratch, samples, metrics):
    DEMO.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for index, sample in enumerate(samples, 1):
        image = check_image(sample)
        truth = ground_truth(sample, image.shape[:2])
        with np.load(scratch/("%02d_torch_cuda.npz" % index),allow_pickle=False) as saved:
            pt = unpack_instances(saved,"display_")
        with np.load(scratch/("%02d_ort_cuda.npz" % index),allow_pickle=False) as saved:
            ort = unpack_instances(saved,"display_")
        grid = np.vstack((np.hstack((panel(image,None,"Original validation image"),
                                    panel(image,truth,"Manual labels (GT)",scores=False))),
                          np.hstack((panel(image,pt,"PyTorch CUDA | display conf=0.25"),
                                     panel(image,ort,"ONNX Runtime CUDA | display conf=0.25")))))
        header = np.full((42,1280,3),(44,36,28),np.uint8)
        cv2.putText(header,"B1 / val%02d  |  GT=%d, PyTorch=%d, ONNX=%d" % (index,len(truth.scores),len(pt.scores),len(ort.scores)),
                    (15,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),1)
        footer=np.full((36,1280,3),245,np.uint8)
        cv2.putText(footer,"AP uses conf=0.001, multi-label; images show conf=0.25. Same physical packages, five validation images.",
                    (14,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,40),1)
        path=DEMO/("val%02d_comparison.png" % index)
        image_write(path,np.vstack((header,grid,footer)))
        if image_read(path).shape!=(886,1280,3):
            raise ValueError("Comparison image dimensions changed")
        artifacts["val%02d" % index]=ref(path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11.8,4.8),layout="constrained")
    labels=["Overall","Sam milk","Yili milk","Luckin cup"]
    for ax,kind,tag in zip(axes,("box","mask"),("B","M")):
        for offset,route,color in zip((-.24,0,.24),ROUTES,("#356cb3","#37a080","#e0a13b")):
            route_metrics=metrics["routes"][route]
            values=[route_metrics["metrics"]["metrics/mAP50-95(%s)" % tag]]
            values += [row[kind+"_mAP50_95"] for row in route_metrics["per_class"]]
            ax.bar(np.arange(4)+offset,values,width=.23,label=route,color=color)
        ax.set_xticks(np.arange(4),labels); ax.set_ylim(0,1.09)
        ax.set_ylabel("mAP50-95"); ax.set_title(kind.capitalize()+" quality (same val5 condition)")
        ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
        for i,value in enumerate([metrics["routes"]["torch_cuda"]["metrics"]["metrics/mAP50-95(%s)" % tag]]+
                                  [r[kind+"_mAP50_95"] for r in metrics["routes"]["torch_cuda"]["per_class"]]):
            ax.text(i,value+.025,"%.4f" % value,ha="center",fontsize=10)
    axes[1].legend(loc="upper right",fontsize=9)
    fig.suptitle("B1 deployment validation | 5 images, 8 instances | no final-test inference",fontsize=13)
    chart=DEMO/"metrics_comparison.png";fig.savefig(chart,dpi=150);plt.close(fig)
    artifacts["metrics_chart"]=ref(chart)
    table=DEMO/"metrics_comparison.csv"
    with table.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(["backend","class","targets","box_P","box_R","box_mAP50","box_mAP50_95",
                                            "mask_P","mask_R","mask_mAP50","mask_mAP50_95"])
        for route in ROUTES:
            for row in metrics["routes"][route]["per_class"]:
                writer.writerow([route,row["name"],row["targets"]]+[row[k] for k in ("box_precision","box_recall","box_mAP50",
                    "box_mAP50_95","mask_precision","mask_recall","mask_mAP50","mask_mAP50_95")])
    artifacts["metrics_csv"]=ref(table)
    sections=''.join('<section><h2>验证图 %d</h2><img src="val%02d_comparison.png" alt="原图、人工标签、PyTorch 和 ONNX 对照"></section>' % (i,i) for i in range(1,6))
    page=DEMO/"index.html"
    page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B1 部署质量对照</title>'
        '<style>body{max-width:1320px;margin:30px auto;background:#f5f7fa;color:#203040;font:16px system-ui;padding:0 20px}img{width:100%;height:auto}section{background:white;padding:18px;margin:24px 0;border-radius:10px}h1{font-size:28px}p{line-height:1.7}</style>'
        '<h1>M5-06：B1 部署质量对照</h1><p>原图 → 人工标注；下方为 PyTorch 与 ONNX CUDA。显示门槛 0.25，AP 评价门槛 0.001；两者用途不同。固定 5 张验证图、8 实例；最终测试集继续封存。</p>'
        '<p>同一包装实物与少量场景的验证结果。历史矩形 / 降采样标签评价不直接与本次原图掩膜结果比较。</p>'
        '<section><h2>总体与各类指标</h2><img src="metrics_comparison.png" alt="三个后端的框与掩膜 mAP 对照"><p><a href="metrics_comparison.csv">下载各类指标 CSV</a></p></section>'
        +sections+'</html>',encoding="utf-8")
    artifacts["gallery"]=ref(page)
    return artifacts


def accept():
    from scripts.check_backend_parity import check_saved as check_previous
    if REPORT.exists() or DEMO.exists():
        raise FileExistsError("Do not overwrite deployment quality evidence; use --check")
    policy,samples=configuration()
    with contextlib.redirect_stdout(sys.stderr):check_previous()
    previous=read_json(PREVIOUS)
    guards={**previous["guarded_files"],**previous["implementation_sha256"],
            **{v["path"]:v["sha256"] for v in previous["artifacts"].values()},
            **{p:file_sha256(ROOT/p) for p in METRIC_SOURCE},PREVIOUS.relative_to(ROOT).as_posix():file_sha256(PREVIOUS)}
    check_hashes(guards);timestamps={p:(ROOT/p).stat().st_mtime_ns for p in guards}
    parent=ROOT/"tmp";parent.mkdir(exist_ok=True)
    runtime={}
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    profile_prefix=ROOT/"artifacts/B1"/("M5-06_cuda_profile_"+stamp)
    with tempfile.TemporaryDirectory(prefix="M5-06_",dir=str(parent)) as temporary:
        scratch=Path(temporary).resolve()
        if not scratch.is_relative_to(parent.resolve()):raise ValueError("Unexpected temporary directory")
        for route in ROUTES:
            print("Inference: "+route+" on frozen val5",flush=True)
            command=["scripts/evaluate_deployment.py","--worker",route,"--scratch",str(scratch)]
            if route=="ort_cuda":command += ["--profile-prefix",str(profile_prefix)]
            runtime[route]=child(command,"__QUALITY_INFERENCE__=")
        print("Evaluating predictions against frozen labels with the common metric core",flush=True)
        metrics=child(["scripts/evaluate_deployment.py","--metric-worker","--scratch",str(scratch)],"__QUALITY_METRICS__=")
        artifacts=render_visuals(scratch,samples,metrics)
    if scratch.exists():raise ValueError("Temporary references were not removed")
    events=Counter(v.get("args",{}).get("provider") for v in read_json(ROOT/runtime["ort_cuda"]["profile"]["path"])
                   if v.get("cat")=="Node" and v.get("args",{}).get("provider"))
    if not events["CUDAExecutionProvider"] or events["CPUExecutionProvider"]:raise ValueError("GPU node execution check failed")
    check_hashes(guards)
    if timestamps!={p:(ROOT/p).stat().st_mtime_ns for p in guards}:raise ValueError("Historical inputs were modified")
    report={"schema_version":1,"task":"M5-06","status":"completed",
        "completed_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),"baseline_id":load_bundle().baseline_id,
        "previous_receipt":ref(PREVIOUS),"configuration":ref(CONFIG),"policy":policy,
        "sources":[{"image_id":s["image_id"],"image":{"path":s["image_path"],"sha256":s["image_sha256"]},
                    "label":{"path":s["label_path"],"sha256":s["label_sha256"]},"split":"val"} for s in samples],
        "runtime_workers":runtime,"evaluation":metrics,"actual_cuda_node_events":dict(events),
        "artifacts":{**artifacts,"cuda_profile":runtime["ort_cuda"]["profile"]},
        "guarded_files":guards,"guarded_files_count":len(guards),"historical_sha256_and_mtime_unchanged":True,
        "implementation_sha256":{p:file_sha256(ROOT/p) for p in IMPLEMENTATION},
        "scope":{"new_photos":0,"label_changes":0,"training":0,"new_model_exports":0,"sealed_test_inference":0,
                 "quality_pool":"val5_only","formal_benchmark":False,"camera_switched_to_onnx":False,
                 "temporary_raw_and_mask_arrays_removed":True,"historical_rect_scores_used_as_drop_reference":False},
        "limitation":"Only five validation images, eight instances and one physical package per class; not broad generalization or final-test results.",
        "progress":{"M5":"6/6","main":"30/54","complete_modules":"4/9","next":"M7-02 camera backend integration or optional M6 TensorRT"}}
    write_json(REPORT,report)
    print(json.dumps({"status":"passed","task":"M5-06","metrics":{r:metrics["routes"][r]["metrics"] for r in ROUTES},
                      "deltas":metrics["deltas"],"guarded_files":len(guards),"cuda_node_events":dict(events)},ensure_ascii=False))


def check_saved():
    from scripts.check_backend_parity import check_saved as check_previous
    report=read_json(REPORT)
    if report["status"]!="completed" or report["task"]!="M5-06":raise ValueError("Expected accepted deployment quality")
    policy,samples=configuration()
    if report["policy"]!=policy or report["baseline_id"]!=load_bundle().baseline_id:raise ValueError("Saved policy changed")
    check_hashes(report["guarded_files"]);check_hashes(report["implementation_sha256"])
    check_hashes({v["path"]:v["sha256"] for v in report["artifacts"].values()})
    routes=report["evaluation"]["routes"]
    for route in ROUTES:
        if routes[route]["images"]!=5 or routes[route]["targets"]!=8:raise ValueError("Evaluation pool drift")
        if [v["image_id"] for v in routes[route]["per_image"]]!=[s["image_id"] for s in samples]:raise ValueError("Image IDs changed")
        if route!="torch_cuda" and metric_deltas(routes["torch_cuda"]["metrics"],routes[route]["metrics"],policy["max_map50_95_drop"])!=report["evaluation"]["deltas"][route]:
            raise ValueError("Saved quality differences do not agree")
    with contextlib.redirect_stdout(sys.stderr):check_previous()
    print(json.dumps({"status":"passed","task":"M5-06","images":5,"guarded_files":len(report["guarded_files"]),
                      "files_written":False,"model_inference":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--check",action="store_true")
    mode.add_argument("--worker",choices=ROUTES);mode.add_argument("--metric-worker",action="store_true")
    parser.add_argument("--scratch");parser.add_argument("--profile-prefix")
    args=parser.parse_args()
    if args.worker or args.metric_worker:
        scratch=Path(args.scratch).resolve()
        if not scratch.is_relative_to((ROOT/"tmp").resolve()) or not scratch.is_dir():raise ValueError("Invalid worker scratch directory")
        if args.worker:inference_worker(args.worker,scratch,args.profile_prefix)
        else:evaluation_worker(scratch)
    elif args.check:check_saved()
    else:accept()


if __name__=="__main__":main()
