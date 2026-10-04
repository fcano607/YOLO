"""GPU E1 preflight and formal training, with frozen inputs and project-local selection."""
import contextlib
import csv
from datetime import datetime
import logging
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

from app.product_data import ROOT, digest, now, read_json, write_json
from app.product_experiment import (MASK_METRIC, experiment_configuration, experiment_guards, fresh_run_paths,
                                    verify_augmentation_acceptance, verify_pool_files, verify_preflight, verify_product_model)
from app.product_training import check_guards, relative, verify_formal_loading
from app.product_trainer import MaskFirstSegmentationTrainer
from scripts.train import Tee, environment_fingerprint, output_shapes


IMPLEMENTATION_PATHS = ("scripts/train.py", "scripts/evaluate.py", "app/product_runner.py",
                        "app/product_experiment.py", "app/product_trainer.py", "app/product_evaluation.py",
                        "app/product_training.py", "scripts/check_training_data.py")


def implementation_hashes():
    return {p: digest(ROOT / p) for p in IMPLEMENTATION_PATHS}


@contextlib.contextmanager
def capture_run_log(log):
    # Upstream LOGGER bound its console stream at import time, before stdout redirection.
    from ultralytics.utils import LOGGER
    handler = logging.StreamHandler(log)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            yield
    finally:
        LOGGER.removeHandler(handler)


def verify_initialization(trainer, pretrained):
    verify_product_model(trainer.model)
    current = trainer.model.state_dict()
    matching = [k for k in current if k in pretrained and current[k].shape == pretrained[k].shape]
    if not matching or any(not torch.equal(current[k].cpu(), pretrained[k]) for k in matching):
        raise ValueError("Compatible pretrained tensors did not transfer exactly")
    return {"head_nc": trainer.model.model[-1].nc, "mask_coefficients": trainer.model.model[-1].nm,
            "names": trainer.model.names, "matching_pretrained_tensors_verified_equal": len(matching),
            "new_or_shape_changed_keys": [k for k in current if k not in matching]}


def finite_losses(items):
    values = {k: float(v.detach()) for k, v in items.items()}
    if not values or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite or missing segmentation losses")
    return values


def run_preflight(config_path, source, name=None):
    config, overrides, loading = experiment_configuration(config_path)
    verify_augmentation_acceptance(config_path, config)
    if not torch.cuda.is_available():
        raise ValueError("E1 preflight requires the configured CUDA device")
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from scripts.check_training_data import check_batch
    stamp = datetime.fromisoformat(now()).strftime("%Y%m%d_%H%M%S")
    probe_args = {**overrides, "project": str(ROOT / "runs/precheck"),
                  "name": name or "probe_" + config["experiment"].replace("-", "") + "_" + stamp, "plots": False, "save": False}
    probe_name, run_path, unused_report, log_path = fresh_run_paths(probe_args)
    report_path = ROOT / config["preflight_report"]
    guards = experiment_guards(config_path, config, loading)
    environment = environment_fingerprint()
    report = {"schema_version": 1, "task": "M3-02" if config["experiment"] == "E1" else "M3_augmentation_control", "status": "running", "started_at": now(),
              "purpose": "product_experiment_gpu_preflight_only", "experiment": config["experiment"], "source": source,
              "config_path": relative(config_path), "config_sha256": digest(config_path),
              "selection": config["selection"], "evaluation": config["evaluation"],
              "planned_configuration": overrides, "probe_overrides": {
                  "project": relative(run_path.parent), "name": probe_name, "plots": False, "save": False},
              "guarded_inputs": guards, "implementation_sha256": implementation_hashes(),
              "data_summary": loading["summary"], "environment_fingerprint_before": environment,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "log": relative(log_path),
              "probe_directory": relative(run_path), "formal_training_run": False,
              "optimizer_steps": 0, "test_model_evaluation_run": False}
    write_json(report_path, report)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log, capture_run_log(log), patch("ultralytics.utils.callbacks.add_integration_callbacks", lambda instance: None):
            init_seeds(overrides["seed"], deterministic=True)
            original = YOLO(str(ROOT / config["weights"]))
            pretrained = {k: v.detach().cpu().clone() for k, v in original.model.state_dict().items()}
            trainer = MaskFirstSegmentationTrainer(overrides=probe_args)
            trainer._setup_train()  # Source-locked setup only; never enters trainer.train().
            report["initialization"] = verify_initialization(trainer, pretrained)
            verify_pool_files(trainer.train_loader.dataset, loading, "train")
            verify_pool_files(trainer.test_loader.dataset, loading, "val")
            initial = {k: v.detach().cpu().clone() for k, v in trainer.model.named_parameters()}
            step_count = [0]
            trainer.optimizer.register_step_post_hook(lambda *args: step_count.__setitem__(0, step_count[0] + 1))
            trainer.model.train()
            records = []
            seen = []
            sample_map = {s["image_id"]: s for s in loading["samples"]}
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            for raw in trainer.train_loader:
                checked = check_batch(raw, sample_map)
                seen.extend(r["image_id"] for r in checked)
                trainer.optimizer.zero_grad(set_to_none=True)
                batch = trainer.preprocess_batch(raw)
                loss, items = trainer.model(batch)
                scalar = loss.sum()
                if not torch.isfinite(scalar):
                    raise ValueError("Non-finite GPU loss")
                scalar.backward()
                gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise ValueError("Missing or non-finite GPU gradients")
                nonzero = sum(bool(g.count_nonzero()) for g in gradients)
                if not nonzero:
                    raise ValueError("GPU backward produced no nonzero gradients")
                records.append({"batch": len(records) + 1, "images": len(raw["im_file"]),
                                "losses": finite_losses(items), "nonzero_gradient_tensors": nonzero,
                                "input_shape": list(batch["img"].shape), "label_checks": checked})
                trainer.loss_items, trainer.loss_names, trainer.loss = items, tuple(items), scalar.detach()
            expected = sorted(s["image_id"] for s in loading["samples"] if s["split"] == "train")
            if sorted(seen) != expected or step_count[0] != 0 or trainer.optimizer.state:
                raise ValueError("Probe skipped sources or unexpectedly performed an optimizer update")
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.epoch = 0
            validation, selected = trainer.validate()  # EMA is the unchanged initialized three-class model.
            from app.product_evaluation import evaluate_model
            standalone = evaluate_model(trainer.ema.ema, config, overrides, loading,
                                        run_path / "standalone_val", plots=False)
            if any(not torch.equal(p.detach().cpu(), initial[k]) for k, p in trainer.model.named_parameters()):
                raise ValueError("Preflight unexpectedly changed trainable parameters")
            with torch.no_grad():
                trainer.model.eval()
                raw_shapes = output_shapes(trainer.model(torch.zeros(1, 3, 640, 640, device="cuda:0")))
            if [1, 39, 8400] not in raw_shapes or [1, 32, 160, 160] not in raw_shapes:
                raise ValueError("Three-class raw output contract changed")
            torch.cuda.synchronize(0)
            report.update(gpu_batches=records, actual_optimizer=type(trainer.optimizer).__name__,
                          optimizer_parameter_groups=[{k: v for k, v in group.items() if k != "params"}
                                                      for group in trainer.optimizer.param_groups],
                          actual_batch=trainer.batch_size, actual_accumulate=trainer.accumulate,
                          actual_amp=trainer.amp, raw_output_shapes=raw_shapes,
                          trainable_parameters_unchanged_in_memory=True,
                          note="Training-mode BatchNorm buffers can change in the disposable probe; pretrained files are never saved or overwritten.",
                          initialized_model_validation={"metrics": validation, "mask_fitness": selected,
                                                        "native_box_plus_mask_fitness": trainer.native_validation_fitness,
                                                        "meaning": "Untrained-head diagnostic only, not E1 accuracy."},
                          standalone_evaluator_check=standalone,
                          planned_budget={"epochs": overrides["epochs"], "batches_per_epoch": len(trainer.train_loader),
                                          "nominal_optimizer_steps": len(trainer.train_loader) * overrides["epochs"],
                                          "warmup_iterations": trainer._get_warmup_iterations(len(trainer.train_loader)),
                                          "accumulate": trainer.accumulate},
                          gpu_probe_wall_seconds=time.perf_counter() - started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
                          resolved_trainer_configuration=vars(trainer.args))
            if list(run_path.rglob("*.pt")):
                raise ValueError("Preflight unexpectedly saved a model checkpoint")
        check_guards(guards)
        check_guards(read_json(ROOT / "reports/experiments/M3-01_products_debug_v1.json")["guarded_inputs"])
        verify_formal_loading()
        after = environment_fingerprint()
        if after != environment:
            raise ValueError("Installed package versions changed during preflight")
        report.update(status="passed", completed_at=now(), guarded_inputs_unchanged=True,
                      environment_fingerprint_after=after, prior_90_guarded_inputs_unchanged=True)
        write_json(report_path, report)
        print(config["experiment"], "preflight PASSED:", relative(report_path), "; optimizer updates=0; this probe did not train")
    except Exception as error:
        report.update(status="failed", completed_at=now(), error=repr(error))
        write_json(report_path, report)
        raise


def run_formal_training(config_path, source, name=None):
    config, overrides, loading = experiment_configuration(config_path)
    verify_augmentation_acceptance(config_path, config)
    preflight = verify_preflight(config_path, config)
    if environment_fingerprint() != preflight["environment_fingerprint_after"]:
        raise ValueError("Runtime packages changed after E1 preflight")
    if not torch.cuda.is_available():
        raise ValueError("E1 requires the prechecked CUDA device")
    chosen, run_path, report_path, log_path = fresh_run_paths(overrides, name)
    overrides["name"] = chosen
    guards = experiment_guards(config_path, config, loading)
    environment = environment_fingerprint()
    report = {"schema_version": 1, "experiment": config["experiment"], "purpose": config["purpose"], "status": "running",
              "started_at": now(), "source": source, "config_path": relative(config_path),
              "config_sha256": digest(config_path), "preflight_sha256": digest(ROOT / config["preflight_report"]),
              "configuration": overrides, "selection_rule": config["selection"], "dataset_summary": loading["summary"],
              "guarded_inputs": guards, "implementation_sha256": implementation_hashes(),
              "environment_fingerprint_before": environment, "run_directory": relative(run_path),
              "log": relative(log_path), "gpu": torch.cuda.get_device_name(0), "test_evaluated": False}
    write_json(report_path, report)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
        with log_path.open("w", encoding="utf-8") as log, capture_run_log(log), patch("ultralytics.utils.callbacks.add_integration_callbacks", lambda instance: None):
            original = YOLO(str(ROOT / config["weights"]))
            pretrained = {k: v.detach().cpu().clone() for k, v in original.model.state_dict().items()}
            trainer = MaskFirstSegmentationTrainer(overrides=overrides)
            snapshot = run_path / "preflight_used.json"
            snapshot.write_bytes((ROOT / config["preflight_report"]).read_bytes())
            if digest(snapshot) != report["preflight_sha256"]:
                raise ValueError("Preflight changed between validation and training setup")
            report["preflight_used_copy"] = {"path": relative(snapshot), "sha256": digest(snapshot)}
            evidence = {"epochs": [], "batches": [], "optimizer_steps": 0}

            def on_start(t):
                evidence["initialization"] = verify_initialization(t, pretrained)
                verify_pool_files(t.train_loader.dataset, loading, "train")
                verify_pool_files(t.test_loader.dataset, loading, "val")
                evidence["optimizer"] = type(t.optimizer).__name__
                evidence["optimizer_parameter_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in t.optimizer.param_groups]
                evidence["accumulate"] = t.accumulate
                def after_step(optimizer, args, kwargs):
                    gradients = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise ValueError("Non-finite optimizer gradients")
                    evidence["optimizer_steps"] += 1
                t.optimizer.register_step_post_hook(after_step)

            def on_batch(t):
                evidence["batches"].append({"epoch": t.epoch + 1, "losses": finite_losses(t.loss_items)})

            def on_epoch(t):
                if t.epoch >= overrides["epochs"]:
                    return
                evidence["epochs"].append({"epoch": t.epoch + 1, "train_losses": finite_losses(t.tloss),
                                           "validation_metrics": {k: float(v) for k, v in t.metrics.items()},
                                           "mask_fitness": float(t.fitness), "best_mask_fitness": float(t.best_fitness),
                                           "best_saved_this_epoch": t.fitness == t.best_fitness, "learning_rates": t.lr})
                check_guards(guards)

            trainer.add_callback("on_train_start", on_start)
            trainer.add_callback("on_train_batch_end", on_batch)
            trainer.add_callback("on_fit_epoch_end", on_epoch)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            trainer.train()
            torch.cuda.synchronize(0)
            evidence["train_wall_seconds"] = time.perf_counter() - started
            evidence["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            if len(evidence["epochs"]) != overrides["epochs"] or evidence["optimizer_steps"] != preflight["planned_budget"]["nominal_optimizer_steps"]:
                raise ValueError("E1 did not finish its declared epoch/update budget")
            if len(evidence["batches"]) != preflight["planned_budget"]["nominal_optimizer_steps"]:
                raise ValueError("E1 skipped or repeated training batches")
            evidence["checkpoints"] = {kind: {"path": relative(p), "sha256": digest(p)}
                                       for kind, p in (("best", trainer.best), ("last", trainer.last))}
            best_epoch = max(evidence["epochs"], key=lambda r: (r["mask_fitness"], r["epoch"]))
            evidence["selected_epoch"] = best_epoch["epoch"]
            evidence["selected_mask_fitness"] = best_epoch["mask_fitness"]
            with (run_path / "results.csv").open(encoding="utf-8") as file:
                rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(file)]
            if len(rows) != overrides["epochs"]:
                raise ValueError("E1 results.csv does not contain one row per epoch")
            evidence["results_csv"] = {"path": relative(run_path / "results.csv"), "sha256": digest(run_path / "results.csv")}
            reloads = []
            for kind, checkpoint in evidence["checkpoints"].items():
                fresh = YOLO(str(ROOT / checkpoint["path"]))
                verify_product_model(fresh)
                fresh.model.to("cuda:0").float().eval()
                with torch.no_grad():
                    shapes = output_shapes(fresh.model(torch.zeros(1, 3, 640, 640, device="cuda:0")))
                if [1, 39, 8400] not in shapes or [1, 32, 160, 160] not in shapes:
                    raise ValueError("Saved E1 raw outputs do not match the three-class contract")
                reloads.append({"checkpoint": kind, "raw_output_shapes": shapes})
            evidence["reload_checks"] = reloads
            report["training"] = evidence
        check_guards(guards)
        verify_formal_loading()
        after = environment_fingerprint()
        if after != environment:
            raise ValueError("Runtime package versions changed during E1")
        report.update(status="completed", completed_at=now(), guarded_inputs_unchanged=True,
                      environment_fingerprint_after=after,
                      conclusion="Declared E1 budget and best/last reload passed; display readiness and error analysis require validation inspection.")
        write_json(report_path, report)
        print(config["experiment"], "completed:", relative(report_path))
    except Exception as error:
        report.update(status="failed", completed_at=now(), error=repr(error))
        write_json(report_path, report)
        raise


def run(config_path, source, preflight=False, name=None):
    if preflight:
        run_preflight(config_path, source, name)
    else:
        run_formal_training(config_path, source, name)
