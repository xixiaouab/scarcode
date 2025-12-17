import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from data import DataFactory
from models import ModelFactory
from utils import (
    set_seed,
    get_logger,
    AverageMeter,
    ProgressMeter,
    CheckpointManager,
    format_logs
)
from utils.losses import JointLoss
from utils.metrics import MetricManager


def parse_args():
    parser = argparse.ArgumentParser(description="ScarElastic Training Entrypoint")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--model_name", type=str, default="scarelastic")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--use_se", action="store_true", default=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")

    return parser.parse_args()


def train_one_epoch(loader, model, criterion, optimizer, scaler, epoch, logger, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_scar_meter = AverageMeter('L_scar', ':.4e')
    loss_deform_meter = AverageMeter('L_deform', ':.4e')
    loss_smooth_meter = AverageMeter('L_smooth', ':.4e')

    model.train()
    end = time.time()

    num_batches = len(loader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, loss_scar_meter, loss_deform_meter, loss_smooth_meter],
        prefix=f"Epoch: [{epoch}]"
    )

    for i, batch_data in enumerate(loader):
        data_time.update(time.time() - end)

        inputs = batch_data["model_input"].to(device)
        targets = {
            "mask": batch_data["mask"].to(device),
            "continuous_target": batch_data["continuous_target"].to(
                device) if "continuous_target" in batch_data else None
        }
        priors = {
            "prior_thickness": batch_data["prior_thickness"].to(device),
            "prior_edge": batch_data["prior_edge"].to(device)
        }

        optimizer.zero_grad()

        with autocast():
            predictions = model(inputs)
            loss_dict = criterion(predictions, targets, priors)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))
        loss_scar_meter.update(loss_dict["loss_scar"].item(), inputs.size(0))
        loss_deform_meter.update(loss_dict["loss_deform"].item(), inputs.size(0))
        loss_smooth_meter.update(loss_dict["loss_smooth"].item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i, logger)

    return losses.avg


def validate(loader, model, criterion, metric_manager, epoch, logger, device):
    losses = AverageMeter('Val Loss', ':.4e')
    metric_manager.reset()
    model.eval()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            inputs = batch_data["model_input"].to(device)
            targets = {
                "mask": batch_data["mask"].to(device),
                "continuous_target": batch_data["continuous_target"].to(
                    device) if "continuous_target" in batch_data else None
            }
            priors = {
                "prior_thickness": batch_data["prior_thickness"].to(device),
                "prior_edge": batch_data["prior_edge"].to(device)
            }

            predictions = model(inputs)
            loss_dict = criterion(predictions, targets, priors)
            losses.update(loss_dict["loss"].item(), inputs.size(0))

            pred_prob = predictions["probability"]
            pred_continuous = predictions["elasticity"]

            metric_manager.update(
                pred_prob,
                targets["mask"],
                continuous_pred=pred_continuous,
                continuous_true=targets["continuous_target"]
            )

    metrics = metric_manager.compute()
    metrics['loss'] = losses.avg

    log_msg = f"Validation Epoch {epoch}: " + format_logs(metrics)
    logger.info(log_msg)

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger("train", os.path.join(args.save_dir, "train.log"))
    logger.info(f"Arguments: {args}")

    config = {
        "data_root": args.data_root,
        "train_json": args.train_json,
        "val_json": args.val_json,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }

    data_factory = DataFactory(config)
    train_loader = data_factory.get_loader("train")
    val_loader = data_factory.get_loader("val")

    model_config = {
        "name": args.model_name,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "base_filters": args.base_filters,
        "depth": args.depth,
        "use_se": args.use_se
    }

    model = ModelFactory.create_model(model_config)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = JointLoss(beta=args.beta, gamma=args.gamma, alpha=args.alpha)
    criterion = criterion.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler()

    metric_manager = MetricManager(metrics=["dice", "hd95", "scs", "pearson"])
    ckpt_manager = CheckpointManager(args.save_dir, model_name=args.model_name, metric_name="dice", mode="max")

    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_manager.load(args.resume, model, optimizer, scheduler, scaler)
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            logger,
            device
        )

        val_metrics = validate(
            val_loader,
            model,
            criterion,
            metric_manager,
            epoch,
            logger,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} completed. LR: {current_lr:.6f}. Train Loss: {train_loss:.4f}")

        ckpt_manager.save(
            model,
            optimizer,
            epoch,
            val_metrics['dice'],
            scheduler,
            scaler
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()