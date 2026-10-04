"""Guard future held-out groups and frozen raw/staged inputs, without touching project photos."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.product_data import digest, write_json
from app import product_training as training


class FrozenDebugSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        data = self.root / "configs/products_debug.yaml"
        data.parent.mkdir(parents=True)
        data.write_text("train: ../data/desktop/images/debug-v1/train\nval: ../data/desktop/images/debug-v1/val\nnames:\n  0: sam_whole_milk\n  1: yili_shuhua\n  2: luckin_cup\n", encoding="utf-8")
        self.inventory_path = self.root / "data/desktop/metadata/dataset_inventory.json"
        self.split_path = self.root / "data/desktop/splits/debug-v1.json"
        self.stack.enter_context(patch.multiple(training, ROOT=self.root, DATA_PATH=data,
                                               INVENTORY_PATH=self.inventory_path, SPLIT_PATH=self.split_path))
        names = {0: "sam_whole_milk", 1: "yili_shuhua", 2: "luckin_cup"}
        classes = {"classes": [{"id": c, "name": n} for c, n in names.items()]}
        write_json(self.root / "data/desktop/metadata/classes.json", classes)
        self.stack.enter_context(patch.object(training, "catalog", return_value=classes))
        samples = []
        for group in (training.TRAIN_GROUP, training.VAL_GROUP):
            for cls in range(3):
                image_id = group + "_" + str(cls)
                sample = {"image_id": image_id, "group_id": group, "status": "reviewed",
                          "instances": [{"class_id": cls}]}
                for field, extension in (("image", "png"), ("label", "txt"), ("review", "json")):
                    relative = "originals/" + image_id + "." + extension
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((field + image_id).encode())
                    sample[field + "_path"], sample[field + "_sha256"] = relative, digest(path)
                draft = self.root / "data/raw/camera/annotations/drafts" / (image_id + ".json")
                draft.parent.mkdir(parents=True, exist_ok=True)
                draft.write_text("draft", encoding="utf-8")
                sample["draft_sha256"] = digest(draft)
                samples.append(sample)
        self.checked = {"samples": samples}
        write_json(self.inventory_path, self.checked)
        self.manifest = training.create_debug_manifest(self.checked, digest(self.inventory_path))
        write_json(self.split_path, self.manifest)
        for sample in self.manifest["samples"]:
            for field in ("image", "label"):
                path = self.root / sample["staged_" + field + "_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((self.root / sample[field + "_path"]).read_bytes())

    def test_related_batches_are_whole_groups_and_never_an_independent_test(self):
        manifest = training.verify_debug_split()
        self.assertFalse(manifest["formal_independent_evaluation"])
        self.assertIsNone(manifest["test"])
        self.assertTrue(set(manifest["summary"]["train"]["capture_groups"]).isdisjoint(
            manifest["summary"]["val"]["capture_groups"]))

    def test_new_capture_group_cannot_silently_enter_training(self):
        checked = deepcopy(self.checked)
        checked["samples"][0]["group_id"] = "new_independent_test"
        with self.assertRaisesRegex(ValueError, "explicit new dataset version"):
            training.create_debug_manifest(checked, digest(self.inventory_path))

    def test_identical_original_cannot_cross_pools(self):
        checked = deepcopy(self.checked)
        checked["samples"][3]["image_sha256"] = checked["samples"][0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "Exact duplicate"):
            training.create_debug_manifest(checked, digest(self.inventory_path))

    def test_staged_edit_is_rejected_and_never_changes_the_raw_original(self):
        sample = self.manifest["samples"][0]
        original = self.root / sample["image_path"]
        (self.root / sample["staged_image_path"]).write_bytes(b"changed training copy")
        self.assertEqual(digest(original), sample["image_sha256"])
        with self.assertRaisesRegex(ValueError, "Staged file differs"):
            training.verify_debug_split()

    def test_review_edit_invalidates_the_frozen_training_version(self):
        sample = self.manifest["samples"][0]
        (self.root / sample["review_path"]).write_text("a new human decision", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            training.verify_debug_split()


if __name__ == "__main__":
    unittest.main()
