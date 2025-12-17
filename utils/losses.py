import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class Gradient3D(nn.Module):
    def __init__(self, penalty: str = "l2"):
        super(Gradient3D, self).__init__()
        self.penalty = penalty
        kernel_x = torch.FloatTensor([[[[-1, 1]]]])
        kernel_y = torch.FloatTensor([[[[-1], [1]]]])
        kernel_z = torch.FloatTensor([[[[-1]], [[1]]]])

        self.register_buffer('kernel_x', kernel_x.unsqueeze(1))
        self.register_buffer('kernel_y', kernel_y.unsqueeze(1))
        self.register_buffer('kernel_z', kernel_z.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_x = F.pad(x, (0, 1, 0, 0, 0, 0), mode="replicate")
        pad_y = F.pad(x, (0, 0, 0, 1, 0, 0), mode="replicate")
        pad_z = F.pad(x, (0, 0, 0, 0, 0, 1), mode="replicate")

        gx = F.conv3d(pad_x, self.kernel_x, stride=1, padding=0)
        gy = F.conv3d(pad_y, self.kernel_y, stride=1, padding=0)
        gz = F.conv3d(pad_z, self.kernel_z, stride=1, padding=0)

        return torch.cat([gx, gy, gz], dim=1)


class ScarRegressionLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(ScarRegressionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.clamp(input, 1e-7, 1 - 1e-7)
        target = torch.clamp(target, 0, 1)

        loss = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SmoothLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(SmoothLoss, self).__init__()
        self.gradient_calculator = Gradient3D()
        self.reduction = reduction

    def forward(self, elasticity_field: torch.Tensor) -> torch.Tensor:
        gradients = self.gradient_calculator(elasticity_field)
        loss = torch.norm(gradients, p=2, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DeformLoss(nn.Module):
    def __init__(self, lambda_topo: float = 0.1):
        super(DeformLoss, self).__init__()
        self.lambda_topo = lambda_topo
        self.gradient_calculator = Gradient3D()

    def _compute_wall_normal(self, thickness_map: torch.Tensor) -> torch.Tensor:
        gradients = self.gradient_calculator(thickness_map)
        norm = torch.norm(gradients, p=2, dim=1, keepdim=True)
        return gradients / (norm + 1e-8)

    def _topology_penalty(self, elasticity: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool3d(elasticity, kernel_size=3, stride=1, padding=1)
        diff = torch.abs(elasticity - pooled)
        return diff.mean()

    def forward(
            self,
            elasticity: torch.Tensor,
            thickness_map: torch.Tensor
    ) -> torch.Tensor:
        grad_elasticity = self.gradient_calculator(elasticity)

        wall_normal = self._compute_wall_normal(thickness_map)

        projection = (grad_elasticity * wall_normal).sum(dim=1, keepdim=True)
        grad_orthogonal = grad_elasticity - projection * wall_normal

        ortho_loss = torch.norm(grad_orthogonal, p=2, dim=1).mean()

        topo_loss = self._topology_penalty(elasticity)

        return ortho_loss + self.lambda_topo * topo_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_flat = input.view(input.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (input_flat * target_flat).sum(dim=1)
        union = input_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class JointLoss(nn.Module):
    def __init__(
            self,
            beta: float = 0.5,
            gamma: float = 0.1,
            alpha: float = 0.6
    ):
        super(JointLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

        self.scar_loss_fn = ScarRegressionLoss()
        self.deform_loss_fn = DeformLoss()
        self.smooth_loss_fn = SmoothLoss()
        self.dice_loss_fn = DiceLoss()

    def forward(
            self,
            predictions: dict,
            targets: dict,
            priors: dict
    ) -> dict:
        pred_prob = predictions["probability"]
        pred_elasticity = predictions["elasticity"]

        target_continuous = targets.get("continuous_target")
        target_binary = targets.get("mask")

        if target_continuous is None:
            target_continuous = target_binary.float()

        loss_scar = self.scar_loss_fn(pred_prob, target_continuous)

        prior_thickness = priors["prior_thickness"]
        loss_deform = self.deform_loss_fn(pred_elasticity, prior_thickness)

        loss_smooth = self.smooth_loss_fn(pred_elasticity)

        loss_dice = self.dice_loss_fn(pred_prob, target_binary)

        loss_regression = self.scar_loss_fn(pred_elasticity, target_continuous)

        total_loss = (
                loss_scar +
                loss_dice +
                self.beta * loss_deform +
                self.gamma * loss_smooth +
                0.5 * loss_regression
        )

        return {
            "loss": total_loss,
            "loss_scar": loss_scar,
            "loss_deform": loss_deform,
            "loss_smooth": loss_smooth,
            "loss_dice": loss_dice
        }