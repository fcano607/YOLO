"""Project-local mask-first trainer; no modifications to the locked upstream source."""
from ultralytics.models.yolo.segment import SegmentationTrainer

from app.product_experiment import mask_fitness


class MaskFirstSegmentationTrainer(SegmentationTrainer):
    def validate(self):
        if self.world_size > 1:
            raise ValueError("The E1 contract was checked for one GPU only")
        metrics = self.validator(self)
        if metrics is None:
            raise ValueError("E1 requires validation every epoch")
        self.native_validation_fitness = metrics.pop("fitness", None)
        fitness = mask_fitness(metrics)
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        # BaseTrainer.save_model writes best.pt on equality, so the latest tied epoch wins.
        return metrics, fitness
