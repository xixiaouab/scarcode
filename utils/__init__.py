import os
import random
import shutil
import logging
import sys
import time
import math
from typing import Dict, Any, Optional, List, Union

import torch
import torch.distributed as dist
import numpy as np


try:
    from .losses import ScarLoss, DeformLoss, SmoothLoss, JointLoss
    from .metrics import DiceMetric, HausdorffDistance95, PearsonCorrelation, StructuralContinuityScore
except ImportError:
    pass

__all__ = [
    "set_seed",
    "get_logger",
    "AverageMeter",
    "ProgressMeter",
    "CheckpointManager",
    "reduce_tensor",
    "all_reduce_mean",
    "gather_tensors",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "time_str",
    "format_logs"
]


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有的 handlers 避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def sync(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count


class ProgressMeter:
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int, logger: logging.Logger = None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class CheckpointManager:
    def __init__(
            self,
            save_dir: str,
            model_name: str = "scarelastic",
            metric_name: str = "dice",
            mode: str = "max"
    ):
        self.save_dir = save_dir
        self.model_name = model_name
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = -float('inf') if mode == 'max' else float('inf')

        os.makedirs(self.save_dir, exist_ok=True)

    def save(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            metric_val: float,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        is_best = False
        if self.mode == 'max':
            if metric_val > self.best_metric:
                self.best_metric = metric_val
                is_best = True
        else:
            if metric_val < self.best_metric:
                self.best_metric = metric_val
                is_best = True

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
        }

        if scheduler:
            state['scheduler'] = scheduler.state_dict()
        if scaler:
            state['scaler'] = scaler.state_dict()

        filename = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}.pth")
        torch.save(state, filename)

        # 保存最新模型链接
        latest_path = os.path.join(self.save_dir, f"{self.model_name}_latest.pth")
        shutil.copyfile(filename, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
            shutil.copyfile(filename, best_path)

        return is_best

    def load(
            self,
            path: str,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            scaler: Optional[torch.cuda.amp.GradScaler] = None,
            device: str = 'cuda'
    ) -> int:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', self.best_metric)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        return start_epoch


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def all_reduce_mean(tensor: Union[int, float, torch.Tensor]) -> float:
    if not is_dist_avail_and_initialized():
        return float(tensor)

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device='cuda')

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor.item()


def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    if not is_dist_avail_and_initialized():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def time_str(t: float) -> str:
    if t < 60:
        return '{:.2f}s'.format(t)
    elif t < 3600:
        return '{:.2f}m'.format(t / 60)
    else:
        return '{:.2f}h'.format(t / 3600)


def format_logs(logs: Dict[str, float]) -> str:
    return " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])