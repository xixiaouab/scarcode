import os
import argparse
import time
import json
import glob
import logging
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from models import ModelFactory
from data import ScarElasticDataset, DataFactory
from data.dataset import InferenceDataset
from utils import (
    get_logger,
    time_str,
    AverageMeter,
    ProgressMeter,
    set_seed
)
from utils.helpers import (
    sliding_window_inference_wrapper,
    remove_small_components,
    save_nifti_result,
    keep_largest_connected_component
)
from utils.metrics import DiceMetric, HausdorffDistance95, StructuralContinuityScore


def parse_args():
    parser = argparse.ArgumentParser(description="ScarElastic Inference Pipeline")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--json_list", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=48)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=4)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--save_elasticity", action="store_true", default=True)
    parser.add_argument("--save_probability", action="store_true", default=True)

    parser.add_argument("--eval_mode", action="store_true")

    return parser.parse_args()


def build_inference_loader(args):
    data_list = []

    if args.json_list:
        with open(args.json_list, "r") as f:
            data = json.load(f)
            if "test" in data:
                data_list = data["test"]
            elif "validation" in data:
                data_list = data["validation"]
            else:
                data_list = data
    else:
        images = sorted(glob.glob(os.path.join(args.data_dir, "*_image.nii.gz")))
        masks = sorted(glob.glob(os.path.join(args.data_dir, "*_mask.nii.gz")))

        for img_path, mask_path in zip(images, masks):
            case_id = os.path.basename(img_path).replace("_image.nii.gz", "")
            item = {
                "image": img_path,
                "mask": mask_path,
                "id": case_id
            }
            label_path = img_path.replace("_image.nii.gz", "_label.nii.gz")
            if os.path.exists(label_path):
                item["label"] = label_path
            data_list.append(item)

    dataset = InferenceDataset(data_list=data_list)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return loader


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("inference", os.path.join(args.output_dir, "inference.log"))

    logger.info(f"Loading model from {args.model_path}")

    model_config = {
        "name": "scarelastic",
        "in_channels": 4,
        "out_channels": 1,
        "pretrained_path": args.model_path,
        "init_weights": False
    }

    model = ModelFactory.create_model(model_config)
    model.to(args.device)
    model.eval()

    test_loader = build_inference_loader(args)
    logger.info(f"Inference dataset size: {len(test_loader)}")

    dice_metric = DiceMetric(threshold=args.threshold)
    hd95_metric = HausdorffDistance95(threshold=args.threshold)
    scs_metric = StructuralContinuityScore(threshold=args.threshold)

    roi_size = (args.roi_z, args.roi_y, args.roi_x)

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(test_loader), [batch_time], prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, batch_data in enumerate(test_loader):
            inputs = batch_data["model_input"].to(args.device)
            original_path = batch_data["original_path"][0]
            case_id = os.path.basename(original_path).split(".")[0].replace("_image", "")

            ref_img = nib.load(original_path)
            affine = ref_img.affine
            header = ref_img.header

            outputs = sliding_window_inference_wrapper(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
                device=args.device
            )

            prob_map = outputs["probability"]
            elasticity_map = outputs["elasticity"]

            prob_np = prob_map.cpu().numpy()[0, 0]
            elast_np = elasticity_map.cpu().numpy()[0, 0]

            binary_mask = (prob_np > args.threshold).astype(np.float32)

            if args.min_cluster_size > 0:
                binary_mask = remove_small_components(binary_mask, min_size=args.min_cluster_size)

            save_subdir = os.path.join(args.output_dir, case_id)
            os.makedirs(save_subdir, exist_ok=True)

            save_nifti_result(
                os.path.join(save_subdir, "pred_mask.nii.gz"),
                binary_mask, affine, header
            )

            if args.save_probability:
                save_nifti_result(
                    os.path.join(save_subdir, "pred_prob.nii.gz"),
                    prob_np, affine, header
                )

            if args.save_elasticity:
                save_nifti_result(
                    os.path.join(save_subdir, "pred_elasticity.nii.gz"),
                    elast_np, affine, header
                )

            if args.eval_mode and "label" in batch_data:  # If label exists and eval requested
                # Note: InferenceDataset currently does not load label by default in __getitem__
                # Assuming user might modify Dataset to include 'label' key for evaluation
                # Or use validation loader logic.
                pass

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i, logger)

    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    main()