import os
import glob
import json
import logging
import argparse
import functools
import multiprocessing
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt, gaussian_laplace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    target_spacing: Tuple[float, float, float] = (1.25, 1.25, 1.25)
    clip_intensity_percentiles: Tuple[float, float] = (0.5, 99.5)
    edge_sigma: float = 1.0
    normalize: bool = True
    output_dtype: type = np.float32


class ImageIO:
    @staticmethod
    def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            affine = img.affine
            return data, affine
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    @staticmethod
    def save_nifti(data: np.ndarray, affine: np.ndarray, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            nifti_img = nib.Nifti1Image(data, affine)
            nib.save(nifti_img, path)
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
            raise


class GeometryProcessor:
    def __init__(self, target_spacing: Tuple[float, float, float]):
        self.target_spacing = np.array(target_spacing)

    def resample(self, data: np.ndarray, affine: np.ndarray, is_mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        current_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        scale_factor = current_spacing / self.target_spacing
        new_shape = np.floor(data.shape * scale_factor).astype(int)

        if is_mask:
            data_resampled = ndimage.zoom(data, scale_factor, order=0, mode='nearest')
        else:
            data_resampled = ndimage.zoom(data, scale_factor, order=3, mode='reflect')

        new_affine = np.copy(affine)
        new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / scale_factor)

        return data_resampled, new_affine


class FeatureGenerator:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def normalize_intensity(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        lower = np.percentile(image, self.config.clip_intensity_percentiles[0])
        upper = np.percentile(image, self.config.clip_intensity_percentiles[1])
        image = np.clip(image, lower, upper)

        if mask is not None and np.sum(mask) > 0:
            roi = image[mask > 0]
            mean, std = np.mean(roi), np.std(roi)
        else:
            mean, std = np.mean(image), np.std(image)

        return (image - mean) / (std + 1e-8)

    def compute_wall_thickness(self, mask: np.ndarray) -> np.ndarray:
        binary_mask = (mask > 0).astype(np.float32)
        if np.sum(binary_mask) == 0:
            return np.zeros_like(mask, dtype=np.float32)

        dist_endo = distance_transform_edt(binary_mask)
        dist_epi = distance_transform_edt(1 - binary_mask)

        thickness_map = dist_endo / (dist_endo + dist_epi + 1e-8)
        thickness_map = thickness_map * binary_mask
        return thickness_map

    def compute_laplacian_edge(self, image: np.ndarray) -> np.ndarray:
        edge_map = gaussian_laplace(image, sigma=self.config.edge_sigma)
        min_val, max_val = np.min(edge_map), np.max(edge_map)
        if max_val - min_val > 1e-8:
            edge_map = (edge_map - min_val) / (max_val - min_val)
        else:
            edge_map = np.zeros_like(edge_map)
        return edge_map


class PreprocessingPipeline:
    def __init__(self, src_root: str, dst_root: str, config: ProcessingConfig):
        self.src_root = src_root
        self.dst_root = dst_root
        self.config = config
        self.io = ImageIO()
        self.geometry = GeometryProcessor(config.target_spacing)
        self.features = FeatureGenerator(config)

    def process_case(self, case_id: str, image_path: str, mask_path: str, label_path: Optional[str] = None) -> Dict:
        try:
            image, affine = self.io.load_nifti(image_path)
            mask, _ = self.io.load_nifti(mask_path)

            label = None
            if label_path and os.path.exists(label_path):
                label, _ = self.io.load_nifti(label_path)

            image_res, affine_res = self.geometry.resample(image, affine, is_mask=False)
            mask_res, _ = self.geometry.resample(mask, affine, is_mask=True)

            if label is not None:
                label_res, _ = self.geometry.resample(label, affine, is_mask=True)
                label_res = (label_res > 0.5).astype(np.float32)
            else:
                label_res = None

            mask_res = (mask_res > 0.5).astype(np.float32)
            if self.config.normalize:
                image_norm = self.features.normalize_intensity(image_res, mask_res)
            else:
                image_norm = image_res

            thickness_map = self.features.compute_wall_thickness(mask_res)
            edge_map = self.features.compute_laplacian_edge(image_norm)

            case_dir = os.path.join(self.dst_root, case_id)
            save_paths = {
                "image": os.path.join(case_dir, "image.nii.gz"),
                "mask": os.path.join(case_dir, "mask.nii.gz"),
                "thickness": os.path.join(case_dir, "prior_thickness.nii.gz"),
                "edge": os.path.join(case_dir, "prior_edge.nii.gz")
            }

            self.io.save_nifti(image_norm.astype(self.config.output_dtype), affine_res, save_paths["image"])
            self.io.save_nifti(mask_res.astype(self.config.output_dtype), affine_res, save_paths["mask"])
            self.io.save_nifti(thickness_map.astype(self.config.output_dtype), affine_res, save_paths["thickness"])
            self.io.save_nifti(edge_map.astype(self.config.output_dtype), affine_res, save_paths["edge"])

            if label_res is not None:
                label_out_path = os.path.join(case_dir, "label.nii.gz")
                self.io.save_nifti(label_res.astype(self.config.output_dtype), affine_res, label_out_path)
                save_paths["label"] = label_out_path

            return {"id": case_id, **save_paths}

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {e}")
            return None


def worker_fn(args):
    pipeline, case_info = args
    return pipeline.process_case(**case_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.25, 1.25, 1.25])
    args = parser.parse_args()

    config = ProcessingConfig(target_spacing=tuple(args.target_spacing))
    pipeline = PreprocessingPipeline(args.src_dir, args.dst_dir, config)

    with open(args.dataset_json, 'r') as f:
        data_manifest = json.load(f)

    all_cases = []
    if "training" in data_manifest:
        for item in data_manifest["training"]:
            case_id = os.path.basename(item["image"]).split(".")[0]
            all_cases.append({
                "case_id": case_id,
                "image_path": os.path.join(args.src_dir, item["image"]),
                "mask_path": os.path.join(args.src_dir, item["mask"]),
                "label_path": os.path.join(args.src_dir, item["label"]) if "label" in item else None
            })

    logger.info(f"Starting preprocessing for {len(all_cases)} cases with {args.num_workers} workers.")

    with multiprocessing.Pool(args.num_workers) as pool:
        process_func = functools.partial(worker_fn)
        args_list = [(pipeline, case) for case in all_cases]
        results = pool.map(worker_fn, args_list)

    processed_list = [r for r in results if r is not None]

    out_json = {
        "training": processed_list,
        "validation": []
    }

    json_save_path = os.path.join(args.dst_dir, "dataset_processed.json")
    with open(json_save_path, 'w') as f:
        json.dump(out_json, f, indent=4)

    logger.info(f"Preprocessing complete. Saved manifest to {json_save_path}")


if __name__ == "__main__":
    main()