"""Frozen debug/formal source splits and training contracts for reviewed product originals."""

from pathlib import Path
import shutil

import yaml

from app.product_data import ROOT, RAW, catalog, digest, read_json, write_json
from app.product_dataset import inventory, load_policy

SPLIT_PATH = ROOT / "data/desktop/splits/debug-v1.json"
INVENTORY_PATH = ROOT / "data/desktop/metadata/dataset_inventory.json"
DATA_PATH = ROOT / "configs/products_debug.yaml"
TRAIN_GROUP = "pilot_20261003130657_more"
VAL_GROUP = "pilot_20261003130657"


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def original_guards(samples):
    guards = {}
    for sample in samples:
        for field in ("image", "review", "label"):
            if sample.get(field + "_path"):
                guards[sample[field + "_path"]] = sample[field + "_sha256"]
        guards["data/raw/camera/annotations/drafts/" + sample["image_id"] + ".json"] = sample["draft_sha256"]
    return guards


def check_guards(guards):
    for path, expected in guards.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input changed: " + path)


def create_debug_manifest(checked, inventory_hash):
    """Never splits a capture group across pools or claims independent evaluation."""
    samples = []
    for sample in checked["samples"]:
        if sample["status"] == "excluded":
            continue
        split = {TRAIN_GROUP: "train", VAL_GROUP: "val"}.get(sample["group_id"])
        if split is None:
            raise ValueError("New capture group requires an explicit new dataset version")
        copied = dict(sample, split=split)
        for field, folder, extension in (("image", "images", ".png"), ("label", "labels", ".txt")):
            copied["staged_" + field + "_path"] = (
                f"data/desktop/{folder}/debug-v1/{split}/{sample['image_id']}{extension}")
        samples.append(copied)
    summary = {}
    for split in ("train", "val"):
        pool = [s for s in samples if s["split"] == split]
        counts = [sum(p["class_id"] == cls for s in pool for p in s["instances"]) for cls in range(3)]
        if not pool or min(counts) == 0:
            raise ValueError("Both debug pools must contain all three classes")
        summary[split] = {"images": len(pool), "class_instances": counts,
                          "negative_images": sum(s["status"] == "negative" for s in pool),
                          "capture_groups": sorted({s["group_id"] for s in pool})}
    if set(summary["train"]["capture_groups"]) & set(summary["val"]["capture_groups"]):
        raise ValueError("Capture group leakage")
    hashes = [s["image_sha256"] for s in samples]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Exact duplicate across debug pools")
    return {"schema_version": 1, "dataset_version": "products-debug-v1", "mapping_version": "products-v1",
            "purpose": "pipeline_debug_only", "formal_independent_evaluation": False, "test": None,
            "warning": "Related capture batches; debug validation is not independent quality evidence.",
            "inventory_path": relative(INVENTORY_PATH), "inventory_sha256": inventory_hash,
            "classes_sha256": digest(ROOT / "data/desktop/metadata/classes.json"),
            "data_yaml_sha256": digest(DATA_PATH), "summary": summary, "samples": samples,
            "augmentation_rule": "Online training augmentation only; all derived views inherit the source pool."}


def build_debug_split():
    checked = read_json(INVENTORY_PATH)
    check_guards(original_guards(checked["samples"]))
    current = inventory(RAW, read_json(ROOT / checked["source_manifest"]))
    if current["samples"] != checked["samples"]:
        raise ValueError("Current reviews differ from the inspected inventory; inspect them again")
    manifest = create_debug_manifest(checked, digest(INVENTORY_PATH))
    if SPLIT_PATH.exists() and read_json(SPLIT_PATH) != manifest:
        raise ValueError("Do not silently overwrite the frozen debug split")
    # Fail before copying when an existing staged file differs or the directory contains extra samples.
    expected_paths = set()
    for sample in manifest["samples"]:
        for field in ("image", "label"):
            destination = ROOT / sample["staged_" + field + "_path"]
            expected_paths.add(destination.resolve())
            if destination.exists() and digest(destination) != sample[field + "_sha256"]:
                raise ValueError("Staged file changed: " + relative(destination))
    for folder, extension in (("images", ".png"), ("labels", ".txt")):
        for path in (ROOT / f"data/desktop/{folder}/debug-v1").rglob("*" + extension):
            if path.resolve() not in expected_paths:
                raise ValueError("Unexpected staged sample: " + relative(path))
    for sample in manifest["samples"]:
        for field in ("image", "label"):
            destination = ROOT / sample["staged_" + field + "_path"]
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / sample[field + "_path"], destination)
    check_guards(original_guards(manifest["samples"]))
    write_json(SPLIT_PATH, manifest)
    return verify_debug_split()


def verify_debug_split():
    manifest = read_json(SPLIT_PATH)
    if (manifest["purpose"] != "pipeline_debug_only" or manifest["formal_independent_evaluation"] is not False or
            manifest["test"] is not None):
        raise ValueError("This entry point only accepts a temporary debug dataset")
    check_guards({manifest["inventory_path"]: manifest["inventory_sha256"],
                  relative(DATA_PATH): manifest["data_yaml_sha256"],
                  "data/desktop/metadata/classes.json": manifest["classes_sha256"],
                  **original_guards(manifest["samples"])})
    expected = create_debug_manifest(read_json(INVENTORY_PATH), manifest["inventory_sha256"])
    if expected != manifest:
        raise ValueError("Frozen assignment or provenance changed")
    data = yaml.safe_load(DATA_PATH.read_text(encoding="utf-8"))
    if data["names"] != {c["id"]: c["name"] for c in catalog()["classes"]} or data.get("test"):
        raise ValueError("Dataset YAML has a different class mapping or an unapproved test pool")
    for split in ("train", "val"):
        directory = (DATA_PATH.parent / data[split]).resolve()
        expected_images = {ROOT / s["staged_image_path"] for s in manifest["samples"] if s["split"] == split}
        if set(directory.glob("*.*")) != expected_images:
            raise ValueError("Training YAML does not address the frozen image pool")
    for sample in manifest["samples"]:
        for field in ("image", "label"):
            if digest(ROOT / sample["staged_" + field + "_path"]) != sample[field + "_sha256"]:
                raise ValueError("Staged file differs from reviewed original")
    return manifest


def debug_configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["schema_version"] != 1 or config["purpose"] != "pipeline_debug_only":
        raise ValueError("Expected a debug training configuration")
    if (ROOT / config["data"]).resolve() != DATA_PATH or (ROOT / config["split_manifest"]).resolve() != SPLIT_PATH:
        raise ValueError("Training configuration must use the frozen debug dataset")
    policy = load_policy(ROOT / config["augment_policy"])
    args = {**policy["train_args"], **config["train_args"]}
    if any(args[k] != v for k, v in policy["train_args"].items()) or args["imgsz"] != policy["imgsz"]:
        raise ValueError("Training overrides the inspected augmentation policy")
    if not 1 <= args["epochs"] <= 3 or args["amp"] is not False or args["workers"] != 0:
        raise ValueError("Debug entry is restricted to 1-3 epochs, FP32 and workers=0")
    args.update(data=str(DATA_PATH), task="segment", project=str(ROOT / args["project"]))
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=args)
    return config, args


def build_formal_splits():
    """Freeze reviewed raw sources and predeclared held-out groups; never trains or stages new copies."""
    intent_path = ROOT / "data/desktop/metadata/holdout_capture_plan.json"
    intent = read_json(intent_path)
    if intent["task"] != "M2-04" or set(intent["groups"]) != {"val", "test"}:
        raise ValueError("Expected the predeclared validation and final-test capture plan")
    if digest(ROOT / "data/desktop/metadata/classes.json") != intent["classes_sha256"]:
        raise ValueError("Product mapping changed during collection")
    pilot = read_json(ROOT / read_json(INVENTORY_PATH)["source_manifest"])
    samples = list(pilot["samples"])
    roles = {group: "train" for group in sorted({s["group_id"] for s in samples})}
    sources = [{"path": read_json(INVENTORY_PATH)["source_manifest"],
                "sha256": digest(ROOT / read_json(INVENTORY_PATH)["source_manifest"])}]
    for role, entry in intent["groups"].items():
        group = entry["group_id"]
        if group in roles or entry["scene_preparation_user_confirmed"] is not True:
            raise ValueError("Held-out group is repeated or its new scene was not confirmed")
        path = RAW / "sessions" / (group + ".json")
        manifest = read_json(path)
        if (manifest["status"] != "captured" or manifest["intended_split"] != role or
                len(manifest["samples"]) != entry["planned_images"] or
                any(s.get("intended_split") != role for s in manifest["samples"])):
            raise ValueError("Capture group differs from its declared role or planned coverage")
        roles[group] = role
        sources.append({"path": relative(path), "sha256": digest(path)})
        samples.extend(manifest["samples"])
    pending = [s["image_id"] for s in samples
               if not (RAW / "annotations/reviewed" / (s["image_id"] + ".json")).exists()]
    if pending:
        raise ValueError("Human review pending: " + ", ".join(pending))
    checked = inventory(RAW, {"group_id": "products_v1_reviewed_sources", "samples": samples})
    for sample in checked["samples"]:
        sample["split"] = "excluded" if sample["status"] == "excluded" else roles[sample["group_id"]]
    cross_candidates = []
    by_id = {s["image_id"]: s for s in checked["samples"]}
    for candidate in checked["similarity_candidates"]:
        a, b = (by_id[i] for i in candidate["images"])
        if a["split"] != b["split"] and "excluded" not in {a["split"], b["split"]}:
            cross_candidates.append(candidate)
    if cross_candidates:
        raise ValueError("Cross-pool similarity requires a human provenance check before freezing: " + str(cross_candidates))
    summary = {}
    for split in ("train", "val", "test"):
        pool = [s for s in checked["samples"] if s["split"] == split]
        counts = [sum(p["class_id"] == cls for s in pool for p in s["instances"]) for cls in range(3)]
        if not pool or min(counts) == 0:
            raise ValueError("Each formal pool must retain reviewed coverage of all three products")
        summary[split] = {"images": len(pool), "class_instances": counts,
                          "negative_images": sum(s["status"] == "negative" for s in pool),
                          "capture_groups": sorted({s["group_id"] for s in pool})}
    checked.update(dataset_version="products-v1", split_status="frozen_reviewed_capture_groups",
                   source_manifests=sources, capture_intent_path=relative(intent_path),
                   capture_intent_sha256=digest(intent_path), classes_sha256=intent["classes_sha256"],
                   independence_note="Different new scenes were confirmed by the user before capture; no exact duplicates or detected cross-pool pHash candidates. Same one physical item per class; broad packaging/general scene independence is not established.")
    manifest = {"schema_version": 1, "dataset_version": "products-v1", "mapping_version": "products-v1",
                "purpose": "formal_small_data_source_split", "status": "frozen_source_assignment",
                "classes_sha256": intent["classes_sha256"], "capture_group_roles": roles, "summary": summary,
                "samples": checked["samples"], "source_manifests": sources,
                "capture_intent_path": relative(intent_path), "capture_intent_sha256": digest(intent_path),
                "validation_use": "Training configuration, weight selection and failure analysis.",
                "test_use": "Final evaluation only; no tuning, model selection or added-training decisions.",
                "training_loader_verified_for_this_version": False,
                "independence_scope": checked["independence_note"]}
    inventory_path = ROOT / "data/desktop/metadata/dataset_inventory_products-v1.json"
    split_path = ROOT / "data/desktop/splits/products-v1.json"
    # The debug inventory and its frozen 14/6 paths remain reproducible.
    for path, value in ((inventory_path, checked), (split_path, manifest)):
        if path.exists() and read_json(path) != value:
            raise ValueError("Do not silently replace a frozen formal dataset version")
    check_guards(original_guards(checked["samples"]))
    for source in sources:
        check_guards({source["path"]: source["sha256"]})
    if not inventory_path.exists():
        write_json(inventory_path, checked)
    if not split_path.exists():
        write_json(split_path, manifest)
    return manifest


FORMAL_SPLIT_PATH = ROOT / "data/desktop/splits/products-v1.json"
FORMAL_DATA_PATH = ROOT / "configs/products_base.yaml"
FORMAL_LOADING_PATH = ROOT / "data/desktop/metadata/products-v1_loading.json"


def formal_loading_plan():
    """Reuse checked debug copies for train; stage only the ten new held-out originals."""
    if not FORMAL_SPLIT_PATH.exists():
        raise ValueError("Freeze the formal source assignment before configuring its loader")
    manifest = build_formal_splits()
    debug = verify_debug_split()
    reusable = {s["image_id"]: s for s in debug["samples"]}
    samples = []
    for original in manifest["samples"]:
        if original["split"] == "excluded":
            continue
        sample = dict(original)
        prior = reusable.get(sample["image_id"])
        if prior and sample["split"] != "train":
            raise ValueError("Old related scenes cannot enter the new held-out pools")
        for field, folder, suffix in (("image", "images", ".png"), ("label", "labels", ".txt")):
            if prior:
                if prior[field + "_sha256"] != sample[field + "_sha256"]:
                    raise ValueError("Reusable debug copy has a different reviewed source")
                sample["staged_" + field + "_path"] = prior["staged_" + field + "_path"]
            else:
                sample["staged_" + field + "_path"] = (
                    f"data/desktop/{folder}/products-v1/{sample['split']}/{sample['image_id']}{suffix}")
        sample["staging_method"] = "reuse_debug_pair" if prior else "copy_reviewed_pair"
        samples.append(sample)
    lists = {}
    for split, name in (("train", "train_base"), ("val", "val_all"), ("test", "test_all")):
        path = ROOT / f"data/desktop/splits/{name}.txt"
        # The locked get_img_files anchors only './' lines to the list file's parent.
        entries = ["./../images/" + s["staged_image_path"].split("/images/", 1)[1]
                   for s in samples if s["split"] == split]
        lists[split] = {"path": relative(path), "content": "\n".join(entries) + "\n"}
    data = {split: "../" + value["path"] for split, value in lists.items()}
    data["names"] = {c["id"]: c["name"] for c in catalog()["classes"]}
    return manifest, samples, lists, data


def build_formal_loading():
    manifest, samples, lists, data = formal_loading_plan()
    # Preflight all existing outputs before any copy; changed versions are never overwritten.
    for sample in samples:
        for field in ("image", "label"):
            path = ROOT / sample["staged_" + field + "_path"]
            if path.exists() and digest(path) != sample[field + "_sha256"]:
                raise ValueError("Staged file changed: " + relative(path))
    for value in lists.values():
        path = ROOT / value["path"]
        if path.exists() and path.read_text(encoding="utf-8") != value["content"]:
            raise ValueError("Frozen image list changed: " + value["path"])
    if FORMAL_DATA_PATH.exists() and yaml.safe_load(FORMAL_DATA_PATH.read_text(encoding="utf-8")) != data:
        raise ValueError("Formal data YAML changed")
    if FORMAL_LOADING_PATH.exists():
        verify_formal_loading()
        return read_json(FORMAL_LOADING_PATH)
    for sample in samples:
        for field in ("image", "label"):
            path = ROOT / sample["staged_" + field + "_path"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / sample[field + "_path"], path)
    for value in lists.values():
        path = ROOT / value["path"]
        if not path.exists():
            path.write_text(value["content"], encoding="utf-8")
    if not FORMAL_DATA_PATH.exists():
        FORMAL_DATA_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    record = {"schema_version": 1, "dataset_version": "products-v1",
              "purpose": "immutable_formal_loader_contract",
              "source_split_path": relative(FORMAL_SPLIT_PATH), "source_split_sha256": digest(FORMAL_SPLIT_PATH),
              "classes_sha256": manifest["classes_sha256"],
              "data_yaml": relative(FORMAL_DATA_PATH), "data_yaml_sha256": digest(FORMAL_DATA_PATH),
              "augment_policy": "configs/augment_products.yaml",
              "augment_policy_sha256": digest(ROOT / "configs/augment_products.yaml"),
              "lists": {split: {"path": v["path"], "sha256": digest(ROOT / v["path"])}
                        for split, v in lists.items()}, "summary": manifest["summary"], "samples": samples,
              "staging_counts": {"reused_image_label_pairs": sum(s["staging_method"] == "reuse_debug_pair" for s in samples),
                                 "new_image_label_pairs": sum(s["staging_method"] == "copy_reviewed_pair" for s in samples)},
              "acceptance_report": "reports/data/M2-05_06_products-v1.json",
              "test_use": manifest["test_use"]}
    write_json(FORMAL_LOADING_PATH, record)
    verify_formal_loading()
    return record


def verify_formal_loading():
    record = read_json(FORMAL_LOADING_PATH)
    manifest, samples, lists, data = formal_loading_plan()
    if (record["purpose"] != "immutable_formal_loader_contract" or record["dataset_version"] != "products-v1" or
            record["source_split_path"] != relative(FORMAL_SPLIT_PATH) or
            record["data_yaml"] != relative(FORMAL_DATA_PATH) or
            record["augment_policy"] != "configs/augment_products.yaml" or
            set(record["lists"]) != {"train", "val", "test"} or
            record["test_use"] != manifest["test_use"] or record["samples"] != samples or
            record["summary"] != manifest["summary"] or record["classes_sha256"] != manifest["classes_sha256"]):
        raise ValueError("Formal loader assignment or mapping changed")
    check_guards({relative(FORMAL_SPLIT_PATH): record["source_split_sha256"],
                  relative(FORMAL_DATA_PATH): record["data_yaml_sha256"],
                  "data/desktop/metadata/classes.json": record["classes_sha256"],
                  record["augment_policy"]: record["augment_policy_sha256"],
                  **original_guards(samples)})
    if yaml.safe_load(FORMAL_DATA_PATH.read_text(encoding="utf-8")) != data:
        raise ValueError("Formal YAML does not address the frozen lists")
    for split, expected in lists.items():
        if record["lists"][split]["path"] != expected["path"]:
            raise ValueError("Formal list location changed")
        path = ROOT / expected["path"]
        if digest(path) != record["lists"][split]["sha256"] or path.read_text(encoding="utf-8") != expected["content"]:
            raise ValueError("Frozen image list changed: " + expected["path"])
    for sample in samples:
        for field in ("image", "label"):
            path = ROOT / sample["staged_" + field + "_path"]
            if digest(path) != sample[field + "_sha256"]:
                raise ValueError("Staged file differs from reviewed original")
    return record
