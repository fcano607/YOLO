"""Control fairness and actual image/polygon alignment under stronger affine augmentation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from app import product_experiment as experiment
from app.product_data import ROOT
from app.product_dataset import apply_preview, inspect_transformed, load_policy, make_pipeline


class AugmentationControlTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "config.yaml"
        self.base = yaml.safe_load((ROOT / "configs/train_baseline.yaml").read_text(encoding="utf-8"))
        self.control = yaml.safe_load((ROOT / "configs/train_augment_control.yaml").read_text(encoding="utf-8"))
        self.base_policy = load_policy(ROOT / self.base["augment_policy"])
        self.control_policy = load_policy(ROOT / self.control["augment_policy"])

    def checked(self, config, policy=None):
        self.path.write_text(yaml.safe_dump(config), encoding="utf-8")
        def read_policy(path):
            return deepcopy(policy or self.control_policy) if Path(path).name == "augment_products_e1a.yaml" else deepcopy(self.base_policy)
        with patch.object(experiment, "controlled_reference"), \
                patch.object(experiment, "baseline_configuration", return_value=(deepcopy(self.base),
                    {**self.base_policy["train_args"], **self.base["train_args"],
                     "model": "locked_weights", "data": self.base["data"]}, {})), \
                patch.object(experiment, "load_policy", side_effect=read_policy):
            return experiment.experiment_configuration(self.path)

    def test_control_changes_only_rotation_scale_and_run_metadata(self):
        config, args, unused = self.checked(self.control)
        self.assertEqual((args["degrees"], args["scale"], args["epochs"], args["nbs"], args["seed"]), (45, .4, 50, 8, 42))
        self.assertFalse(args["resume"])
        self.assertEqual(args["model"], "locked_weights")

    def test_control_rejects_budget_optimizer_seed_data_and_evaluation_changes(self):
        for section, key, value in (("train_args", "epochs", 100), ("train_args", "seed", 43),
                                    ("train_args", "lr0", .01), ("train_args", "resume", True),
                                    ("evaluation", "split", "test")):
            changed = deepcopy(self.control)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "may change only"):
                self.checked(changed)
        changed = deepcopy(self.control)
        changed["data"] = "configs/products_debug.yaml"
        with self.assertRaisesRegex(ValueError, "may change only"):
            self.checked(changed)

    def test_control_rejects_hidden_hsv_or_flip_changes(self):
        for key, value in (("hsv_v", .5), ("fliplr", .5), ("degrees", 90)):
            changed = deepcopy(self.control_policy)
            changed["train_args"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.checked(self.control, changed)

    def test_visual_acceptance_is_required_before_preflight(self):
        config = deepcopy(self.control)
        report = {"status": "passed", "visual_review": {"accepted": False},
                  "config_sha256": "sha", "policy_sha256": "sha", "loading_contract_sha256": "sha",
                  "guarded_inputs": {}, "implementation_sha256": {}}
        with patch.object(experiment, "read_json", return_value=report), patch.object(experiment, "digest", return_value="sha"), \
                patch.object(experiment, "check_guards"):
            with self.assertRaisesRegex(ValueError, "visual acceptance"):
                experiment.verify_augmentation_acceptance(self.path, config)
            report["visual_review"]["accepted"] = True
            experiment.verify_augmentation_acceptance(self.path, config)
            report["policy_sha256"] = "changed"
            with self.assertRaises(ValueError):
                experiment.verify_augmentation_acceptance(self.path, config)

    def test_rotation_and_scale_transform_pixels_with_their_polygons(self):
        policy = deepcopy(self.control_policy)
        policy["train_args"].update(hsv_s=0., hsv_v=0.)
        pipeline = make_pipeline(policy)
        image = np.zeros((360, 640, 3), np.uint8)
        points = np.array([[.4, .35], [.6, .35], [.6, .65], [.4, .65]])
        cv2.fillPoly(image, [np.rint(points * [640, 360]).astype(np.int32)], (0, 255, 0))
        for seed in (42, 43, 44):
            result = apply_preview(pipeline, image, [{"class_id": 1, "points": points.tolist()}], 640, seed)
            self.assertEqual(inspect_transformed(result, [1])["retained_class_ids"], [1])
            marker = result["img"][..., 1] > 128
            mask = np.zeros((640, 640), np.uint8)
            cv2.fillPoly(mask, [np.rint(result["instances"].segments[0]).astype(np.int32)], 1)
            # Padding has G=114; marker has G=255, so the colored object is measured independently.
            intersection = np.logical_and(marker, mask).sum()
            union = np.logical_or(marker, mask).sum()
            with self.subTest(seed=seed):
                self.assertGreater(intersection / union, .97)
