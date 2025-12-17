import os
import sys
import argparse
import time
import shutil
import logging
import traceback
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from data import DataFactory
from models import ModelFactory
from utils import (
    set_seed,
    get_logger,
    AverageMeter,
    CheckpointManager,
    format_logs
)
from utils.losses import JointLoss
from utils.metrics import MetricManager


class ScarElasticTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(args.seed)
        self._init_dirs()
        self.logger = get_logger("train", os.path.join(args.save_dir, "train.log"))
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

        self.logger.info(f"Configuration: {vars(args)}")

        self._init_data()
        self._init_model()
        self._init_optimization()

        self.metric_manager = MetricManager(metrics=["dice", "hd95", "scs", "pearson"])
        self.ckpt_manager = CheckpointManager(
            args.save_dir,
            model_name=args.model_name,
            metric_name="dice",
            mode="max"
        )

        self.start_epoch = 0
        if args.resume:
            self._resume_checkpoint(args.resume)

        self.early_stopping_patience = args.patience
        self.no_improve_epochs = 0

    def _init_dirs(self):
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.save_dir, "tensorboard"), exist_ok=True)

    def _init_data(self):
        config = {
            "data_root": self.args.data_root,
            "train_json": self.args.train_json,
            "val_json": self.args.val_json,
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "seed": self.args.seed
        }
        factory = DataFactory(config)
        self.train_loader = factory.get_loader("train")
        self.val_loader = factory.get_loader("val")
        self.logger.info(f"Data Loaded: Train={len(self.train_loader)}, Val={len(self.val_loader)}")

    def _init_model(self):
        config = {
            "name": self.args.model_name,
            "in_channels": self.args.in_channels,
            "out_channels": self.args.out_channels,
            "base_filters": self.args.base_filters,
            "depth": self.args.depth,
            "use_se": self.args.use_se,
            "dropout": self.args.dropout
        }
        self.model = ModelFactory.create_model(config)
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model {self.args.model_name} initialized. Trainable Params: {params:,}")

    def _init_optimization(self):
        self.criterion = JointLoss(
            beta=self.args.beta,
            gamma=self.args.gamma,
            alpha=self.args.alpha
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.args.epochs,
            T_mult=1,
            eta_min=1e-6
        )

        self.scaler = GradScaler()

    def _resume_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}...")
        self.start_epoch = self.ckpt_manager.load(
            path, self.model, self.optimizer, self.scheduler, self.scaler
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = {
            "loss": AverageMeter("Loss"),
            "scar": AverageMeter("Scar"),
            "deform": AverageMeter("Deform"),
            "smooth": AverageMeter("Smooth")
        }

        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            inputs = batch["model_input"].to(self.device)
            targets = {
                "mask": batch["mask"].to(self.device),
                "continuous_target": batch.get("continuous_target", batch["mask"]).to(self.device)
            }
            priors = {
                "prior_thickness": batch["prior_thickness"].to(self.device),
                "prior_edge": batch["prior_edge"].to(self.device)
            }

            self.optimizer.zero_grad()

            with autocast():
                preds = self.model(inputs)
                loss_dict = self.criterion(preds, targets, priors)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.args.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = inputs.size(0)
            meters["loss"].update(loss.item(), bs)
            meters["scar"].update(loss_dict["loss_scar"].item(), bs)
            meters["deform"].update(loss_dict["loss_deform"].item(), bs)
            meters["smooth"].update(loss_dict["loss_smooth"].item(), bs)

            if i % self.args.log_interval == 0:
                self.logger.info(
                    f"Epoch [{epoch}][{i}/{len(self.train_loader)}] "
                    f"Loss: {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})"
                )

        self.writer.add_scalar("Train/Loss", meters["loss"].avg, epoch)
        self.writer.add_scalar("Train/Loss_Scar", meters["scar"].avg, epoch)
        self.writer.add_scalar("Train/Loss_Deform", meters["deform"].avg, epoch)
        self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]['lr'], epoch)

        return meters["loss"].avg

    def validate(self, epoch):
        self.model.eval()
        self.metric_manager.reset()
        val_loss = AverageMeter("ValLoss")

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["model_input"].to(self.device)
                targets = {
                    "mask": batch["mask"].to(self.device),
                    "continuous_target": batch.get("continuous_target", batch["mask"]).to(self.device)
                }
                priors = {
                    "prior_thickness": batch["prior_thickness"].to(self.device),
                    "prior_edge": batch["prior_edge"].to(self.device)
                }

                preds = self.model(inputs)
                loss_dict = self.criterion(preds, targets, priors)
                val_loss.update(loss_dict["loss"].item(), inputs.size(0))

                self.metric_manager.update(
                    preds["probability"],
                    targets["mask"],
                    continuous_pred=preds["elasticity"],
                    continuous_true=targets["continuous_target"]
                )

        metrics = self.metric_manager.compute()
        metrics["loss"] = val_loss.avg

        self.logger.info(f"Validation Epoch {epoch}: {format_logs(metrics)}")

        for k, v in metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch)

        return metrics

    def run(self):
        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                self.logger.info(f"Started Epoch {epoch}")

                self.train_one_epoch(epoch)
                metrics = self.validate(epoch)

                self.scheduler.step()

                is_best = self.ckpt_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    metrics["dice"],
                    self.scheduler,
                    self.scaler
                )

                if is_best:
                    self.no_improve_epochs = 0
                    self.logger.info(f"New Best Model! Dice: {metrics['dice']:.4f}")
                else:
                    self.no_improve_epochs += 1

                if self.no_improve_epochs >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs.")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Saving emergency checkpoint...")
            self.ckpt_manager.save(
                self.model, self.optimizer, epoch, 0.0, self.scheduler, self.scaler
            )
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.writer.close()
            self.logger.info("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="ScarElastic Trainer")

    # Path Arguments
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./work_dirs/experiment_1")
    parser.add_argument("--resume", type=str, default=None)

    # Model Arguments
    parser.add_argument("--model_name", type=str, default="scarelastic")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--use_se", action="store_true", default=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training Arguments
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    # Loss Arguments (Paper Params)
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for Deform Loss")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight for Smooth Loss")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for Elasticity Fusion")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = ScarElasticTrainer(args)
    trainer.run()