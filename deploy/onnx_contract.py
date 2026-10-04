"""Validate the static B1 raw-output interface, separately from detection quality."""
import ast

import numpy as np


def validate_interface(inputs, outputs, embedded, op_types, names):
    expected_inputs = [{"name": "images", "shape": [1, 3, 640, 640], "onnx_dtype": 1}]
    expected_outputs = [{"name": "output0", "shape": [1, 39, 8400], "onnx_dtype": 1},
                        {"name": "output1", "shape": [1, 32, 160, 160], "onnx_dtype": 1}]
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError("Expected static batch1 / 640 / FP32 three-class raw outputs")
    if "NonMaxSuppression" in op_types:
        raise ValueError("NMS must remain outside the exported graph")
    expected_names = {int(k): v for k, v in names.items()}
    if (embedded.get("task") != "segment" or ast.literal_eval(embedded.get("names", "{}")) != expected_names or
            ast.literal_eval(embedded.get("imgsz", "[]")) != [640, 640] or
            embedded.get("batch") != "1" or embedded.get("end2end") != "False"):
        raise ValueError("Embedded class mapping or segmentation metadata differs from B1")


def inspect_graph(path, names):
    import onnx
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)

    def tensor_info(value):
        return {"name": value.name, "shape": [d.dim_param or d.dim_value
                for d in value.type.tensor_type.shape.dim], "onnx_dtype": value.type.tensor_type.elem_type}

    def operators(current):
        values = []
        for node in current.node:
            values.append(node.op_type)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    values.extend(operators(attribute.g))
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        values.extend(operators(nested))
        return values

    inputs, outputs = [tensor_info(v) for v in graph.graph.input], [tensor_info(v) for v in graph.graph.output]
    embedded = {v.key: v.value for v in graph.metadata_props}
    op_types = operators(graph.graph)
    validate_interface(inputs, outputs, embedded, op_types, names)
    opsets = {v.domain or "ai.onnx": v.version for v in graph.opset_import}
    if opsets.get("ai.onnx") != 17 or graph.ir_version > 10:
        raise ValueError("Expected opset17 and ONNX Runtime compatible IR<=10")
    if any(v.data_type == onnx.TensorProto.FLOAT16 for v in graph.graph.initializer):
        raise ValueError("FP16 weights are not allowed in the FP32 baseline export")
    return {"inputs": inputs, "outputs": outputs, "metadata": embedded, "ir_version": graph.ir_version,
            "opsets": opsets, "node_count": len(graph.graph.node), "nms_embedded": False,
            "checker_full_check": True}


def select_smoke_sources(loading, image_ids):
    by_id = {s["image_id"]: s for s in loading["samples"]}
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("Smoke image IDs must be nonempty and unique")
    selected = []
    for image_id in image_ids:
        sample = by_id.get(image_id)
        if sample is None or sample["split"] != "val":
            raise ValueError("Export smoke checks may use only the frozen val pool, never sealed test")
        selected.append(sample)
    return selected


def compare_raw(reference, actual, tolerance):
    if len(reference) != len(actual) or len(actual) != 2:
        raise ValueError("Expected both candidate and prototype outputs")
    records = []
    for index, (ref, pred) in enumerate(zip(reference, actual)):
        if ref.shape != pred.shape or ref.dtype != np.float32 or pred.dtype != np.float32:
            raise ValueError("Raw output shape or FP32 dtype mismatch")
        if not np.isfinite(ref).all() or not np.isfinite(pred).all():
            raise ValueError("Non-finite raw outputs")
        error = np.abs(ref.astype(np.float64) - pred.astype(np.float64))
        bound = tolerance["atol"] + tolerance["rtol"] * np.abs(ref.astype(np.float64))
        outside = int(np.count_nonzero(error > bound))
        records.append({"name": "output" + str(index), "shape": list(pred.shape), "dtype": str(pred.dtype),
                        "max_abs": float(error.max()), "mean_abs": float(error.mean()),
                        "p99_abs": float(np.quantile(error, .99)), "outside_tolerance": outside,
                        "allclose": outside == 0})
    return records
