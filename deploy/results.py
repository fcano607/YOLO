"""Frame results shared by independent ONNX and PyTorch raw inference."""
from dataclasses import dataclass

from deploy.postprocess import ProductInstances
from deploy.preprocess import LetterboxGeometry


@dataclass
class InferenceResult:
    instances: ProductInstances
    geometry: LetterboxGeometry
    class_names: dict
    backend: dict
    timings_ms: dict

    def to_dict(self):
        instances = self.instances
        return {"original_shape_hw": list(self.geometry.original_shape_hw), "backend": self.backend,
                "timings_ms": self.timings_ms, "counts": instances.counts,
                "instances": [{"candidate_index": int(index), "class_id": int(cls),
                               "label": self.class_names[int(cls)], "confidence": float(score),
                               "box_xyxy": box.tolist(), "mask_pixels": int(mask.sum())}
                              for index, cls, score, box, mask in zip(instances.candidate_indices,
                                  instances.class_ids, instances.scores, instances.boxes_xyxy, instances.masks)]}
