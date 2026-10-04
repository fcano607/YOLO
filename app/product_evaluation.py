"""Frozen validation-only model evaluation; final test inference is deliberately not part of M3."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.product_data import ROOT
from app.product_experiment import mask_fitness, verify_pool_files, verify_product_model


def evaluate_model(model, config, overrides, loading, save_dir, plots=True):
    from ultralytics.models.yolo.segment import SegmentationValidator
    verify_product_model(model)
    ev = config["evaluation"]
    if ev["split"] != "val" or overrides["split"] != "val":
        raise ValueError("M3 evaluation is restricted to the frozen validation pool")
    args = {"data": str(ROOT / config["data"]), "task": "segment", "split": "val",
            "imgsz": overrides["imgsz"], "batch": overrides["batch"] * 2, "device": 0,
            "workers": 0, "quantize": None, "conf": ev["metric_conf"], "iou": ev["nms_iou"],
            "max_det": ev["max_det"], "augment": False, "plots": plots, "save_json": False,
            "overlap_mask": True, "mask_ratio": 4, "rect": True, "fraction": 1.0}
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    validator = SegmentationValidator(args=args, save_dir=save_dir)
    with patch("ultralytics.utils.callbacks.add_integration_callbacks", lambda instance: None):
        metrics = validator(model=model)
    verify_pool_files(validator.dataloader.dataset, loading, "val")
    per_class = []
    for i, row in enumerate(validator.metrics.summary()):
        result = {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
        # The upstream summary's generic mAP fields are box-only; report mask AP explicitly.
        box = validator.metrics.box.class_result(i)
        mask = validator.metrics.seg.class_result(i)
        result.update({"Class-ID": int(validator.metrics.ap_class_index[i]),
                       "Box-mAP50": float(box[2]), "Box-mAP50-95": float(box[3]),
                       "Mask-mAP50": float(mask[2]), "Mask-mAP50-95": float(mask[3])})
        per_class.append(result)
    return {"split": "val", "images": len(validator.dataloader.dataset),
            "image_ids": [Path(p).stem for p in validator.dataloader.dataset.im_files],
            "metrics": {k: float(v) for k, v in metrics.items()}, "mask_fitness": mask_fitness(metrics),
            "per_class": per_class,
            "settings": args, "test_images_evaluated": 0}
