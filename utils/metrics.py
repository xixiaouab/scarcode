import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Optional, Union, Tuple


class BaseMetric:
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val: Union[float, torch.Tensor, np.ndarray]):
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.item()
        self.values.append(val)

    def compute(self) -> float:
        if len(self.values) == 0:
            return 0.0
        if self.reduction == 'mean':
            return sum(self.values) / len(self.values)
        elif self.reduction == 'sum':
            return sum(self.values)
        return self.values[-1]


class DiceMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = (y_pred > self.threshold).float()
        y_true = (y_true > 0.5).float()

        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            pred_flat = y_pred[i].view(-1)
            true_flat = y_true[i].view(-1)

            intersection = (pred_flat * true_flat).sum()
            union = pred_flat.sum() + true_flat.sum()

            score = (2. * intersection + self.smooth) / (union + self.smooth)
            self.update(score)

        return self.compute()


class IoUMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = (y_pred > self.threshold).float()
        y_true = (y_true > 0.5).float()

        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            pred_flat = y_pred[i].view(-1)
            true_flat = y_true[i].view(-1)

            intersection = (pred_flat * true_flat).sum()
            total = (pred_flat + true_flat).sum()
            union = total - intersection

            score = (intersection + self.smooth) / (union + self.smooth)
            self.update(score)

        return self.compute()


class SensitivityMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = (y_pred > self.threshold).float()
        y_true = (y_true > 0.5).float()

        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            pred_flat = y_pred[i].view(-1)
            true_flat = y_true[i].view(-1)

            intersection = (pred_flat * true_flat).sum()
            true_sum = true_flat.sum()

            score = (intersection + self.smooth) / (true_sum + self.smooth)
            self.update(score)

        return self.compute()


class SpecificityMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = (y_pred > self.threshold).float()
        y_true = (y_true > 0.5).float()

        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            pred_flat = y_pred[i].view(-1)
            true_flat = y_true[i].view(-1)

            tn = ((1 - pred_flat) * (1 - true_flat)).sum()
            fp = (pred_flat * (1 - true_flat)).sum()

            score = (tn + self.smooth) / (tn + fp + self.smooth)
            self.update(score)

        return self.compute()


class HausdorffDistance95(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def _compute_surface_distances(self, result: np.ndarray, reference: np.ndarray,
                                   voxel_spacing: Tuple[float, float, float] = (1.25, 1.25, 1.25)):
        result = result.astype(bool)
        reference = reference.astype(bool)

        if not np.any(result):
            return np.inf
        if not np.any(reference):
            return np.inf

        result_border = result ^ distance_transform_edt(result, sampling=voxel_spacing) < 1.0
        reference_border = reference ^ distance_transform_edt(reference, sampling=voxel_spacing) < 1.0

        dt_ref = distance_transform_edt(~reference_border, sampling=voxel_spacing)
        dt_res = distance_transform_edt(~result_border, sampling=voxel_spacing)

        sds_result = dt_ref[result_border]
        sds_reference = dt_res[reference_border]

        return np.concatenate([sds_result, sds_reference])

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                 spacing: Tuple[float, float, float] = (1.25, 1.25, 1.25)):
        y_pred_np = y_pred.detach().cpu().numpy() > self.threshold
        y_true_np = y_true.detach().cpu().numpy() > 0.5

        batch_size = y_pred_np.shape[0]
        for i in range(batch_size):
            try:
                surface_distances = self._compute_surface_distances(y_pred_np[i, 0], y_true_np[i, 0], spacing)
                if np.isinf(surface_distances).any():
                    hd95 = 100.0
                else:
                    hd95 = np.percentile(surface_distances, 95)
                self.update(hd95)
            except Exception:
                self.update(100.0)

        return self.compute()


class PearsonCorrelation(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            pred_flat = y_pred[i].view(-1)
            true_flat = y_true[i].view(-1)

            vx = pred_flat - torch.mean(pred_flat)
            vy = true_flat - torch.mean(true_flat)

            numerator = torch.sum(vx * vy)
            denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))

            score = numerator / (denominator + 1e-8)
            self.update(score.item())

        return self.compute()


class StructuralContinuityScore(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def _count_components(self, volume: np.ndarray) -> int:
        labeled, num_features = label(volume)
        return num_features

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy() > self.threshold
        y_true_np = y_true.detach().cpu().numpy() > 0.5

        batch_size = y_pred_np.shape[0]
        for i in range(batch_size):
            n_pred = self._count_components(y_pred_np[i, 0])
            n_true = self._count_components(y_true_np[i, 0])

            diff = abs(n_pred - n_true)
            denom = max(n_pred, n_true)

            if denom == 0:
                score = 1.0
            else:
                score = 1.0 - (diff / denom)

            self.update(score)

        return self.compute()


class MetricManager:
    def __init__(self, metrics: List[str] = ["dice", "iou", "hd95", "scs", "pearson"]):
        self.metrics_map = {
            "dice": DiceMetric(),
            "iou": IoUMetric(),
            "sensitivity": SensitivityMetric(),
            "specificity": SpecificityMetric(),
            "hd95": HausdorffDistance95(),
            "scs": StructuralContinuityScore(),
            "pearson": PearsonCorrelation()
        }
        self.active_metrics = {k: self.metrics_map[k] for k in metrics if k in self.metrics_map}

    def reset(self):
        for metric in self.active_metrics.values():
            metric.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, continuous_pred: Optional[torch.Tensor] = None,
               continuous_true: Optional[torch.Tensor] = None):
        for name, metric in self.active_metrics.items():
            if name == "pearson":
                if continuous_pred is not None and continuous_true is not None:
                    metric(continuous_pred, continuous_true)
                else:
                    metric(y_pred, y_true.float())
            else:
                metric(y_pred, y_true)

    def compute(self) -> Dict[str, float]:
        return {name: metric.compute() for name, metric in self.active_metrics.items()}