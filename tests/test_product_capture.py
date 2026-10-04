"""Capture-plan roles and quota fixtures; never opens a camera or touches real photos."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.product_data import ROOT, review_task, write_json
from app.product_data import digest
from app import product_training

spec = importlib.util.spec_from_file_location("product_capture_entry", ROOT / "scripts/annotate_product_drafts.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CapturePlanTests(unittest.TestCase):
    def formal_fixture(self, task_root):
        raw = task_root / "data/raw/camera"
        pilot_path = raw / "sessions/pilot.json"
        write_json(pilot_path, {"samples": [{"image_id": "original", "group_id": "pilot"}]})
        inventory = task_root / "data/desktop/metadata/dataset_inventory.json"
        write_json(inventory, {"source_manifest": pilot_path.relative_to(task_root).as_posix()})
        classes = task_root / "data/desktop/metadata/classes.json"
        write_json(classes, {"fixture": True})
        groups = {}
        for role in ("val", "test"):
            group = "new_" + role
            groups[role] = {"group_id": group, "planned_images": 5, "scene_preparation_user_confirmed": True}
            write_json(raw / "sessions" / (group + ".json"), {"status": "captured", "intended_split": role,
                "samples": [{"image_id": group + "_" + str(i), "group_id": group, "intended_split": role} for i in range(5)]})
        write_json(task_root / "data/desktop/metadata/holdout_capture_plan.json",
                   {"task": "M2-04", "classes_sha256": digest(classes), "groups": groups})
        return raw, inventory

    def test_five_prompts_cover_three_singles_mixed_and_empty(self):
        self.assertEqual([s["expected_class_ids"] for s in capture.capture_plan("holdout-five")],
                         [[0], [1], [2], [0, 1, 2], []])
        self.assertEqual([len(capture.capture_plan(p)) for p in ("six", "remaining", "full")], [6, 14, 20])

    def test_review_bundle_keeps_heldout_task_despite_missing_bundle_role(self):
        manifest = {"samples": [{"intended_split": "val"}, {"intended_split": "test"}]}
        self.assertEqual(review_task(manifest), "M2-04")
        self.assertEqual(review_task({"samples": [{"image_id": "pilot"}]}), "M2-02")

    def test_exhausted_budget_blocks_before_image_write(self):
        with tempfile.TemporaryDirectory() as folder:
            task_root = Path(folder).resolve()
            raw = task_root / "data/raw/camera"
            write_json(raw / "sessions/initial.json", {"samples": [{"image_id": "original"}]})
            write_json(raw / "sessions/more.json", {"samples": [{"image_id": "extra_" + str(i)} for i in range(30)]})
            write_json(task_root / "data/desktop/metadata/sampling_budget.json", {"baseline_image_ids": ["original"]})
            with patch.multiple(capture, ROOT=task_root, RAW=raw), patch.object(capture, "image_write") as writer:
                self.assertEqual(capture.capture_budget()["remaining_additional_real_images"], 0)
                with self.assertRaisesRegex(ValueError, "exhausted"):
                    capture.add_sample({"group_id": "new", "samples": []}, np.zeros((10, 10, 3), np.uint8),
                                       {"expected_class_ids": [0]})
                writer.assert_not_called()
                self.assertFalse((raw / "frames").exists())

    def test_declared_roles_and_capture_prompts_cannot_replace_human_review(self):
        with tempfile.TemporaryDirectory() as folder:
            task_root = Path(folder).resolve()
            raw, inventory = self.formal_fixture(task_root)
            with patch.multiple(product_training, ROOT=task_root, RAW=raw, INVENTORY_PATH=inventory):
                with self.assertRaisesRegex(ValueError, "Human review pending"):
                    product_training.build_formal_splits()
            self.assertFalse((task_root / "data/desktop/splits/products-v1.json").exists())

    def test_changed_heldout_role_blocks_formal_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            task_root = Path(folder).resolve()
            raw, inventory = self.formal_fixture(task_root)
            path = raw / "sessions/new_test.json"
            from app.product_data import read_json
            changed = read_json(path)
            changed["intended_split"] = "train"
            write_json(path, changed)
            with patch.multiple(product_training, ROOT=task_root, RAW=raw, INVENTORY_PATH=inventory):
                with self.assertRaisesRegex(ValueError, "declared role"):
                    product_training.build_formal_splits()
            self.assertFalse((task_root / "data/desktop/splits/products-v1.json").exists())


if __name__ == "__main__":
    unittest.main()
