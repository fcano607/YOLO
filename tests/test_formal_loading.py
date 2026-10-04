"""Prevent held-out leakage, altered labels and accidental duplicate staging in the formal entry."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import product_training as training
from app.product_data import digest, read_json, write_json


class FormalLoadingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        stack = ExitStack()
        self.addCleanup(stack.close)
        classes = {"classes": [{"id": c, "name": "product_" + str(c)} for c in range(3)]}
        write_json(self.root / "data/desktop/metadata/classes.json", classes)
        policy = self.root / "configs/augment_products.yaml"
        policy.parent.mkdir(parents=True)
        policy.write_text("train_args: {}\n", encoding="utf-8")
        samples, debug = [], []
        for split in ("train", "val", "test"):
            for cls in range(3):
                image_id = split + "_" + str(cls)
                sample = {"image_id": image_id, "group_id": split + "_scene", "split": split,
                          "status": "reviewed", "instances": [{"class_id": cls}]}
                for field, suffix in (("image", ".png"), ("label", ".txt"), ("review", ".json")):
                    path = self.root / "originals" / (image_id + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((field + image_id).encode())
                    sample[field + "_path"] = path.relative_to(self.root).as_posix()
                    sample[field + "_sha256"] = digest(path)
                draft = self.root / "data/raw/camera/annotations/drafts" / (image_id + ".json")
                draft.parent.mkdir(parents=True, exist_ok=True)
                draft.write_text("draft", encoding="utf-8")
                sample["draft_sha256"] = digest(draft)
                samples.append(sample)
                if split == "train":
                    copied = deepcopy(sample)
                    # A former debug val directory is a formal training source, selected by the explicit list.
                    old_role = "val" if cls == 0 else "train"
                    for field, folder, suffix in (("image", "images", ".png"), ("label", "labels", ".txt")):
                        relative = f"data/desktop/{folder}/debug-v1/{old_role}/{image_id}{suffix}"
                        path = self.root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes((self.root / sample[field + "_path"]).read_bytes())
                        copied["staged_" + field + "_path"] = relative
                    debug.append(copied)
        self.manifest = {"samples": samples, "summary": {s: {"images": 3} for s in ("train", "val", "test")},
                         "classes_sha256": digest(self.root / "data/desktop/metadata/classes.json"),
                         "test_use": "Final evaluation only"}
        split_path = self.root / "data/desktop/splits/products-v1.json"
        write_json(split_path, self.manifest)
        stack.enter_context(patch.multiple(training, ROOT=self.root, FORMAL_SPLIT_PATH=split_path,
                                          FORMAL_DATA_PATH=self.root / "configs/products_base.yaml",
                                          FORMAL_LOADING_PATH=self.root / "data/desktop/metadata/products-v1_loading.json"))
        stack.enter_context(patch.object(training, "catalog", return_value=classes))
        stack.enter_context(patch.object(training, "build_formal_splits", return_value=self.manifest))
        stack.enter_context(patch.object(training, "verify_debug_split", return_value={"samples": debug}))
        self.record = training.build_formal_loading()

    def test_roles_follow_explicit_lists_and_existing_pairs_are_reused(self):
        self.assertEqual(self.record["staging_counts"], {"reused_image_label_pairs": 3, "new_image_label_pairs": 6})
        train = (self.root / self.record["lists"]["train"]["path"]).read_text(encoding="utf-8")
        self.assertIn("images/debug-v1/val/train_0.png", train)
        self.assertNotIn("test_", train)
        before = {p.relative_to(self.root).as_posix(): digest(p) for p in self.root.rglob("*") if p.is_file()}
        training.build_formal_loading()
        after = {p.relative_to(self.root).as_posix(): digest(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_final_test_cannot_be_added_to_train_even_if_list_hash_is_updated(self):
        path = self.root / self.record["lists"]["train"]["path"]
        path.write_text(path.read_text(encoding="utf-8") + "./../images/products-v1/test/test_0.png\n", encoding="utf-8")
        record_path = training.FORMAL_LOADING_PATH
        changed = read_json(record_path)
        changed["lists"]["train"]["sha256"] = digest(path)
        write_json(record_path, changed)
        with self.assertRaisesRegex(ValueError, "Frozen image list changed"):
            training.verify_formal_loading()

    def test_modified_staged_label_is_rejected_and_original_preserved(self):
        sample = next(s for s in self.record["samples"] if s["split"] == "test")
        (self.root / sample["staged_label_path"]).write_text("a wrong target", encoding="utf-8")
        self.assertEqual(digest(self.root / sample["label_path"]), sample["label_sha256"])
        with self.assertRaisesRegex(ValueError, "Staged file differs"):
            training.verify_formal_loading()

    def test_redirected_yaml_in_loader_metadata_is_rejected(self):
        changed = read_json(training.FORMAL_LOADING_PATH)
        changed["data_yaml"] = "configs/products_debug.yaml"
        write_json(training.FORMAL_LOADING_PATH, changed)
        with self.assertRaisesRegex(ValueError, "assignment or mapping changed"):
            training.verify_formal_loading()


if __name__ == "__main__":
    unittest.main()
