import os
import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from skimage import measure, morphology
from monai.inferers import sliding_window_inference


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.numpy()
    return tensor


def min_max_normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val + eps)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc.astype(np.float32)


def remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    mask_bool = mask > 0.5
    cleaned = morphology.remove_small_objects(mask_bool, min_size=min_size)
    return cleaned.astype(np.float32)


def compute_patch_counts(
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        strides: Tuple[int, int, int]
) -> np.ndarray:
    d, h, w = img_size
    pd, ph, pw = patch_size
    sd, sh, sw = strides

    count_map = np.zeros(img_size)

    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                count_map[z:z + pd, y:y + ph, x:x + pw] += 1

    return count_map


def sliding_window_inference_wrapper(
        inputs: torch.Tensor,
        roi_size: Tuple[int, int, int],
        sw_batch_size: int,
        predictor: torch.nn.Module,
        overlap: float = 0.5,
        mode: str = "gaussian",
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    def predictor_wrapper(x):
        return predictor(x)["probability"]

    def elasticity_wrapper(x):
        return predictor(x)["elasticity"]

    prob_map = sliding_window_inference(
        inputs,
        roi_size,
        sw_batch_size,
        predictor_wrapper,
        overlap=overlap,
        mode=mode,
        padding_mode="constant",
        cval=0.0
    )

    elasticity_map = sliding_window_inference(
        inputs,
        roi_size,
        sw_batch_size,
        elasticity_wrapper,
        overlap=overlap,
        mode=mode,
        padding_mode="constant",
        cval=0.0
    )

    return {"probability": prob_map, "elasticity": elasticity_map}


class ImageSaver:
    def __init__(self, output_dir: str, cmap: str = 'jet'):
        self.output_dir = output_dir
        self.cmap = plt.get_cmap(cmap)
        os.makedirs(self.output_dir, exist_ok=True)

    def _apply_colormap(self, image: np.ndarray) -> np.ndarray:
        norm_img = min_max_normalize(image)
        colored = self.cmap(norm_img)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def _overlay(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255),
                 alpha: float = 0.4) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask_bool = mask > 0.5
        overlay = image.copy()

        for i in range(3):
            overlay[:, :, i] = np.where(
                mask_bool,
                (1 - alpha) * image[:, :, i] + alpha * color[i],
                image[:, :, i]
            )
        return overlay

    def save_volume_slices(
            self,
            case_id: str,
            image: np.ndarray,
            pred_mask: np.ndarray,
            gt_mask: Optional[np.ndarray] = None,
            elasticity: Optional[np.ndarray] = None,
            axis: int = 2,
            slice_skip: int = 1
    ):
        img_vol = tensor_to_numpy(image)
        pred_vol = tensor_to_numpy(pred_mask)
        gt_vol = tensor_to_numpy(gt_mask) if gt_mask is not None else None
        elast_vol = tensor_to_numpy(elasticity) if elasticity is not None else None

        img_vol = min_max_normalize(img_vol) * 255
        img_vol = img_vol.astype(np.uint8)

        num_slices = img_vol.shape[axis]

        for i in range(0, num_slices, slice_skip):
            if axis == 0:
                slc_img = img_vol[i, :, :]
                slc_pred = pred_vol[i, :, :]
                slc_gt = gt_vol[i, :, :] if gt_vol is not None else None
                slc_elast = elast_vol[i, :, :] if elast_vol is not None else None
            elif axis == 1:
                slc_img = img_vol[:, i, :]
                slc_pred = pred_vol[:, i, :]
                slc_gt = gt_vol[:, i, :] if gt_vol is not None else None
                slc_elast = elast_vol[:, i, :] if elast_vol is not None else None
            else:
                slc_img = img_vol[:, :, i]
                slc_pred = pred_vol[:, :, i]
                slc_gt = gt_vol[:, :, i] if gt_vol is not None else None
                slc_elast = elast_vol[:, :, i] if elast_vol is not None else None

            if np.sum(slc_pred) == 0 and (slc_gt is None or np.sum(slc_gt) == 0):
                continue

            canvas = []

            rgb_img = cv2.cvtColor(slc_img, cv2.COLOR_GRAY2RGB)
            canvas.append(rgb_img)

            pred_overlay = self._overlay(slc_img, slc_pred, color=(0, 255, 0))
            if slc_gt is not None:
                gt_overlay = self._overlay(slc_img, slc_gt, color=(0, 0, 255))
                combined_overlay = self._overlay(gt_overlay, slc_pred, color=(0, 255, 0))
                canvas.append(combined_overlay)
            else:
                canvas.append(pred_overlay)

            if slc_elast is not None:
                elast_heatmap = self._apply_colormap(slc_elast)
                canvas.append(elast_heatmap)

            final_img = np.concatenate(canvas, axis=1)

            save_name = f"{case_id}_axis{axis}_slice{i:03d}.png"
            cv2.imwrite(os.path.join(self.output_dir, save_name), final_img)


class MetricLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.history = {}

    def log(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        if phase not in self.history:
            self.history[phase] = []

        entry = {'epoch': epoch, **metrics}
        self.history[phase].append(entry)

        with open(self.log_file, 'a') as f:
            msg = f"Phase: {phase} | Epoch: {epoch} | "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            f.write(msg + "\n")

    def get_best_epoch(self, metric: str = 'dice', phase: str = 'val', mode: str = 'max') -> Tuple[int, float]:
        if phase not in self.history:
            return 0, 0.0

        records = self.history[phase]
        values = [r[metric] for r in records]

        if mode == 'max':
            best_val = max(values)
            best_idx = values.index(best_val)
        else:
            best_val = min(values)
            best_idx = values.index(best_val)

        return records[best_idx]['epoch'], best_val


def save_nifti_result(
        save_path: str,
        data: np.ndarray,
        affine: np.ndarray,
        header: Optional[Dict] = None
):
    import nibabel as nib

    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    nifti_img = nib.Nifti1Image(data, affine, header=header)
    nib.save(nifti_img, save_path)


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    shape = list(labels.shape)
    shape[1] = num_classes
    labels_one_hot = torch.zeros(shape).to(labels.device)
    labels_one_hot.scatter_(1, labels.long(), 1)
    return labels_one_hot


def compute_uncertainty(probabilities: torch.Tensor) -> torch.Tensor:
    return -1.0 * torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)