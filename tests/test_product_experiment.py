"""Exercise E1 checkpoint selection, sealed test data, serialization and stale-run guards."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from app import product_evaluation as evaluation
from app import product_experiment as experiment
from app.product_data import ROOT, digest, write_json
from app.product_trainer import MaskFirstSegmentationTrainer


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_reloaded_torch_network_with_task_and_public_wrapper_both_verify(self):
        from ultralytics.engine.model import Model
        names = {0: "product_0", 1: "product_1", 2: "product_2"}
        head = torch.nn.Module()
        head.nc, head.proto = 3, torch.nn.Identity()
        net = torch.nn.Module()
        net.names, net.task = names, "segment"
        net.model = torch.nn.Sequential(head)
        wrapper = object.__new__(Model)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = net
        with patch.object(experiment, "catalog", return_value={"classes": [
                {"id": k, "name": v} for k, v in names.items()]}):
            experiment.verify_product_model(net)
            experiment.verify_product_model(wrapper)
            head.nc = 80
            with self.assertRaisesRegex(ValueError, "three-product"):
                experiment.verify_product_model(net)

    def test_mask_winner_overrides_native_box_plus_mask_and_latest_tie_wins(self):
        trainer = object.__new__(MaskFirstSegmentationTrainer)
        trainer.world_size, trainer.best_fitness = 1, None
        saved_epochs = []
        # Native selection would choose epoch 1 (0.90 + 0.20), not epoch 3.
        for epoch, (box, mask) in enumerate(((0.90, 0.20), (0.20, 0.60), (0.10, 0.60), (0.85, 0.15)), 1):
            metrics = {"metrics/mAP50-95(B)": box, experiment.MASK_METRIC: mask, "fitness": box + mask}
            trainer.validator = lambda _: deepcopy(metrics)
            remaining, fitness = trainer.validate()
            self.assertEqual(fitness, mask)
            self.assertNotIn("fitness", remaining)
            self.assertEqual(trainer.native_validation_fitness, box + mask)
            # Source-locked BaseTrainer.save_model saves best.pt on this equality.
            if trainer.best_fitness == fitness:
                saved_epochs.append(epoch)
        self.assertEqual(saved_epochs, [1, 2, 3])
        self.assertEqual(saved_epochs[-1], 3)
        self.assertEqual(trainer.best_fitness, 0.60)

    def test_bad_mask_metrics_do_not_replace_best(self):
        trainer = object.__new__(MaskFirstSegmentationTrainer)
        trainer.world_size, trainer.best_fitness = 1, 0.5
        for values in ({"fitness": 0.9}, {experiment.MASK_METRIC: float("nan")},
                       {experiment.MASK_METRIC: True}, {experiment.MASK_METRIC: 1.01}):
            with self.subTest(values=values):
                trainer.validator = lambda _: deepcopy(values)
                with self.assertRaises(ValueError):
                    trainer.validate()
                self.assertEqual(trainer.best_fitness, 0.5)

    def test_formal_config_rejects_test_pool_aug_drift_and_debug_resume(self):
        base = yaml.safe_load((ROOT / "configs/train_baseline.yaml").read_text(encoding="utf-8"))
        loading = {"data_yaml": base["data"], "acceptance_report": base["data_acceptance"],
                   "augment_policy": base["augment_policy"]}
        accepted = {"status": "formal_quality_and_loader_accepted",
                    "loading_contract_sha256": base["weights_sha256"],
                    "training_loader_verified_for_this_version": True}
        policy = {"imgsz": 640, "train_args": {"degrees": 5.0}}
        with patch.object(experiment, "verify_formal_loading", return_value=loading), \
                patch.object(experiment, "read_json", return_value=accepted), \
                patch.object(experiment, "digest", return_value=base["weights_sha256"]), \
                patch.object(experiment, "load_policy", return_value=policy):
            path = self.root / "config.yaml"
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            experiment.baseline_configuration(path)
            for field, value, expected in (("split", "test", "declared val pool"),
                                            ("degrees", 45, "augmentation policy"),
                                            ("resume", True, "original pretrained")):
                with self.subTest(field=field):
                    changed = deepcopy(base)
                    changed["train_args"][field] = value
                    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        experiment.baseline_configuration(path)

    def test_preflight_rejects_stale_config_or_implementation(self):
        config_path = self.root / "config.yaml"
        config_path.write_text("fixed config", encoding="utf-8")
        script = self.root / "train.py"
        script.write_text("fixed implementation", encoding="utf-8")
        report = {"status": "passed", "config_sha256": digest(config_path), "guarded_inputs": {},
                  "implementation_sha256": {"train.py": digest(script)}}
        config = {"preflight_report": "preflight.json"}
        write_json(self.root / "preflight.json", report)
        with patch.object(experiment, "ROOT", self.root), patch.object(experiment, "check_guards"):
            experiment.verify_preflight(config_path, config)
            script.write_text("changed implementation", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                experiment.verify_preflight(config_path, config)
            config_path.write_text("changed config", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact formal configuration"):
                experiment.verify_preflight(config_path, config)

    def test_existing_run_report_or_log_cannot_be_overwritten(self):
        args = {"project": str(self.root / "runs/train"), "name": "E1"}
        with patch.object(experiment, "ROOT", self.root):
            unused, run, report, log = experiment.fresh_run_paths(args)
            for path in (run, report, log):
                with self.subTest(path=path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("prior evidence", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Run already exists"):
                        experiment.fresh_run_paths(args)
                    self.assertEqual(path.read_text(encoding="utf-8"), "prior evidence")
                    path.unlink()

    def test_evaluator_rejects_test_before_validator_or_output_creation(self):
        with patch.object(evaluation, "verify_product_model"), \
                patch("ultralytics.models.yolo.segment.SegmentationValidator") as validator:
            for cfg_split, arg_split in (("test", "val"), ("val", "test")):
                with self.subTest(cfg_split=cfg_split, arg_split=arg_split):
                    with self.assertRaisesRegex(ValueError, "validation pool"):
                        evaluation.evaluate_model(object(), {"evaluation": {"split": cfg_split}},
                                                  {"split": arg_split}, {}, self.root / "output")
            validator.assert_not_called()
            self.assertFalse((self.root / "output").exists())

    def test_evaluator_numpy_class_counts_can_be_saved_as_json(self):
        validator = Mock()
        validator.return_value = {experiment.MASK_METRIC: np.float64(0.5)}
        validator.dataloader.dataset = type("Dataset", (), {"__len__": lambda _: 1})()
        validator.dataloader.dataset.im_files = ["/val_1.jpg"]
        validator.metrics.summary.return_value = [{"Class": "product", "Images": np.int64(1),
                                                   "Instances": np.int64(2), "Mask-mAP50-95": np.float64(0.5)}]
        validator.metrics.ap_class_index = np.array([2])
        validator.metrics.box.class_result.return_value = (0.9, 0.8, 0.7, 0.6)
        validator.metrics.seg.class_result.return_value = (0.9, 0.8, 0.65, 0.5)
        config = {"data": "configs/products_base.yaml", "evaluation": {
            "split": "val", "metric_conf": 0.001, "nms_iou": 0.7, "max_det": 300}}
        args = {"split": "val", "imgsz": 640, "batch": 8}
        with patch.object(evaluation, "verify_product_model"), patch.object(evaluation, "verify_pool_files"), \
                patch("ultralytics.models.yolo.segment.SegmentationValidator", return_value=validator):
            result = evaluation.evaluate_model(object(), config, args, {}, self.root / "output")
        serialized = json.dumps(result)
        self.assertEqual(json.loads(serialized)["per_class"][0]["Instances"], 2)
        self.assertEqual(result["per_class"][0]["Class-ID"], 2)
        self.assertEqual(result["per_class"][0]["Box-mAP50-95"], 0.6)
        self.assertEqual(result["per_class"][0]["Mask-mAP50-95"], 0.5)
        self.assertEqual(result["test_images_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
