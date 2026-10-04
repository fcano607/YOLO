"""Meaningful baseline guards: evidence pairing, held-out contamination and tampering."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.product_data import write_json
from scripts import freeze_product_baseline as freeze


class ProductBaselineTests(unittest.TestCase):
    def setUp(self):
        checkpoint = {"path": "best.pt", "sha256": "saved-best"}
        self.loading = {"samples": [{"image_id": role + str(i), "split": role}
                                    for role, count in (("train", 20), ("val", 5), ("test", 5))
                                    for i in range(count)], "summary": {"train": 20, "val": 5, "test": 5}}
        metric = {"split": "val", "image_ids": ["val" + str(i) for i in range(5)], "images": 5,
                  "test_images_evaluated": 0, "settings": {"conf": .001}, "metrics": {"mask_map": .78}}
        self.training = {"status": "completed", "experiment": "E1-A", "configuration": {"batch": 8},
                         "selection_rule": {"metric": "mask"}, "test_evaluated": False,
                         "dataset_summary": self.loading["summary"], "training": {
                             "epochs": list(range(50)), "optimizer_steps": 150, "selected_epoch": 50,
                             "checkpoints": {"best": checkpoint}}}
        self.evaluation = {"status": "completed", "checkpoint": checkpoint, "split": "val",
                           "test_evaluated": False, "metric_evaluation": metric}
        self.live = {"weights": checkpoint, "predict": {"conf": .25, "quantize": 32}}
        options = {**self.live["predict"], "save": False, "save_txt": False,
                   "show": False, "verbose": False, "classes": None}
        self.camera = {"status": "completed", "weights": checkpoint, "camera_confirmed_by_user": True,
                       "test_images_inferred": 0, "final_runtime_configuration_check": {"passed": True, "predict": options}}
        self.manifest = {"baseline_id": freeze.BASELINE_ID, "status": "frozen",
                         "selected_model": {"checkpoint": checkpoint}, "training": {
                             "epochs": 50, "optimizer_steps": 150, "configuration": self.training["configuration"],
                             "selection_rule": self.training["selection_rule"]},
                         "data": {"summary": self.loading["summary"], "test_model_evaluated": False,
                                  "image_ids": {role: [s["image_id"] for s in self.loading["samples"] if s["split"] == role]
                                                for role in ("train", "val", "test")}},
                         "official_evaluation": metric, "live_runtime": {"predict": options}}

    def validate(self, manifest=None, evaluation=None):
        freeze.validate_contract(manifest or self.manifest, self.training, evaluation or self.evaluation,
                                 self.camera, self.loading, self.live)

    def test_wrong_checkpoint_and_training_budget_are_rejected(self):
        self.validate()
        for field in ("checkpoint", "budget"):
            wrong = deepcopy(self.manifest)
            if field == "checkpoint":
                wrong["selected_model"]["checkpoint"]["path"] = "last_other_run.pt"
            else:
                wrong["training"]["optimizer_steps"] = 300
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(wrong)

    def test_final_test_cannot_be_relabelled_as_validation(self):
        wrong = deepcopy(self.evaluation)
        wrong["metric_evaluation"]["image_ids"][0] = "test0"
        with self.assertRaisesRegex(ValueError, "final test"):
            self.validate(evaluation=wrong)

    def test_metric_and_live_confidence_have_distinct_contracts(self):
        for section in ("official_evaluation", "live_runtime"):
            wrong = deepcopy(self.manifest)
            if section == "official_evaluation":
                wrong[section]["settings"]["conf"] = .25
            else:
                wrong[section]["predict"]["conf"] = .001
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.validate(wrong)

    def test_changed_weight_or_manifest_fails_readonly_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best.pt").write_bytes(b"frozen weights")
            with patch.object(freeze, "ROOT", root), patch("app.product_training.ROOT", root):
                manifest = {"baseline_id": freeze.BASELINE_ID, "guarded_files": {
                    "best.pt": freeze.file_reference("best.pt")["sha256"]}}
                write_json(root / freeze.MANIFEST, manifest)
                receipt = {"status": "completed", "baseline_id": freeze.BASELINE_ID,
                           "manifest": freeze.file_reference(freeze.MANIFEST)}
                freeze.verify_frozen_files(manifest, receipt)
                (root / "best.pt").write_bytes(b"replaced weights")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    freeze.verify_frozen_files(manifest, receipt)
                (root / freeze.MANIFEST).write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "acceptance receipt"):
                    freeze.verify_frozen_files(manifest, receipt)


if __name__ == "__main__":
    unittest.main()
