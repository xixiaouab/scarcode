import os
import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, gaussian_laplace, gaussian_filter
from typing import Dict, List, Optional, Union, Tuple


class ScarElasticDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            datalist_json: str,
            transform=None,
            phase: str = "train",
            cache_rate: float = 0.0,
            continuous_sigma: float = 1.0,
            edge_sigma: float = 1.0,
            normalize_method: str = "mask_region"
    ):
        self.data_root = data_root
        self.transform = transform
        self.phase = phase
        self.cache_rate = cache_rate
        self.continuous_sigma = continuous_sigma
        self.edge_sigma = edge_sigma
        self.normalize_method = normalize_method

        with open(datalist_json, "r") as f:
            self.data_list = json.load(f)[phase]

        self.cache = {}
        if self.cache_rate > 0:
            self._init_cache()

    def _init_cache(self):
        num_to_cache = int(len(self.data_list) * self.cache_rate)
        for i in range(num_to_cache):
            self._load_and_cache(i)

    def _load_and_cache(self, index: int):
        item = self.data_list[index]
        data = self._load_single_case(item)
        self.cache[index] = data

    def _load_nifti(self, path: str) -> np.ndarray:
        full_path = os.path.join(self.data_root, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        img = nib.load(full_path)
        return img.get_fdata().astype(np.float32)

    def _normalize_intensity(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.normalize_method == "mask_region":
            roi_voxels = image[mask > 0]
            if len(roi_voxels) > 0:
                mean = np.mean(roi_voxels)
                std = np.std(roi_voxels)
                return (image - mean) / (std + 1e-8)
            else:
                return (image - np.mean(image)) / (np.std(image) + 1e-8)
        elif self.normalize_method == "z_score":
            return (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image

    def _compute_thickness_map(self, mask: np.ndarray) -> np.ndarray:
        binary_mask = (mask > 0).astype(np.float32)
        dist_endo = distance_transform_edt(binary_mask)
        dist_epi = distance_transform_edt(1 - binary_mask)
        thickness_map = dist_endo / (dist_endo + dist_epi + 1e-8)
        return thickness_map * binary_mask

    def _compute_edge_map(self, image: np.ndarray) -> np.ndarray:
        edge_map = gaussian_laplace(image, sigma=self.edge_sigma)
        edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map) + 1e-8)
        return edge_map

    def _generate_continuous_target(self, scar_mask: np.ndarray) -> np.ndarray:
        return gaussian_filter(scar_mask.astype(np.float32), sigma=self.continuous_sigma)

    def _load_single_case(self, item: Dict) -> Dict:
        image_path = item["image"]
        mask_path = item["mask"]
        label_path = item["label"] if "label" in item else None

        image = self._load_nifti(image_path)
        mask = self._load_nifti(mask_path)
        mask[mask > 0] = 1

        image = self._normalize_intensity(image, mask)

        thickness_map = self._compute_thickness_map(mask)
        edge_map = self._compute_edge_map(image)

        data = {
            "image": image,
            "mask": mask,
            "prior_thickness": thickness_map,
            "prior_edge": edge_map,
            "image_path": image_path
        }

        if label_path:
            label = self._load_nifti(label_path)
            label[label > 0] = 1
            data["label"] = label
            data["continuous_target"] = self._generate_continuous_target(label)

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index in self.cache:
            data = self.cache[index].copy()
        else:
            data = self._load_single_case(self.data_list[index])

        if self.transform:
            data = self.transform(data)

        if isinstance(data, dict):
            if "image" in data and isinstance(data["image"], torch.Tensor):
                c, h, w, d = data["image"].shape
                if c == 1:
                    full_input = [data["image"]]

                    if "mask" in data:
                        full_input.append(data["mask"])
                    if "prior_thickness" in data:
                        full_input.append(data["prior_thickness"])
                    if "prior_edge" in data:
                        full_input.append(data["prior_edge"])

                    data["model_input"] = torch.cat(full_input, dim=0)

        return data


class InferenceDataset(Dataset):
    def __init__(self, data_list: List[Dict], transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        img_path = data["image"]

        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(data["mask"]).get_fdata().astype(np.float32)
        mask[mask > 0] = 1

        roi_voxels = image[mask > 0]
        if len(roi_voxels) > 0:
            mean = np.mean(roi_voxels)
            std = np.std(roi_voxels)
            image = (image - mean) / (std + 1e-8)

        thickness_map = distance_transform_edt(mask) / (
                    distance_transform_edt(mask) + distance_transform_edt(1 - mask) + 1e-8)
        thickness_map = thickness_map * mask

        edge_map = gaussian_laplace(image, sigma=1.0)
        edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map) + 1e-8)

        sample = {
            "image": image,
            "mask": mask,
            "prior_thickness": thickness_map,
            "prior_edge": edge_map,
            "original_path": img_path
        }

        if self.transform:
            sample = self.transform(sample)

        full_input = [sample["image"], sample["mask"], sample["prior_thickness"], sample["prior_edge"]]
        sample["model_input"] = torch.cat(full_input, dim=0)

        return sample