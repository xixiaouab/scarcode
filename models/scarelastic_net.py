import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional
from .backbone import UNetBackbone


class RegressionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(RegressionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels // 2, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ScarElasticNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 1,
            base_filters: int = 16,
            depth: int = 4,
            use_se: bool = True,
            spatial_dims: int = 3,
            dropout_rate: float = 0.0
    ):
        super(ScarElasticNet, self).__init__()

        self.backbone = UNetBackbone(
            in_channels=in_channels,
            base_filters=base_filters,
            depth=depth,
            dropout_rate=dropout_rate
        )

        final_filters = self.backbone.final_filters

        self.elasticity_head = RegressionHead(
            in_channels=final_filters,
            out_channels=out_channels,
            dropout_rate=dropout_rate
        )

        self.probability_head = RegressionHead(
            in_channels=final_filters,
            out_channels=out_channels,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        elasticity_field = self.elasticity_head(features)
        scar_probability = self.probability_head(features)

        return {
            "elasticity": elasticity_field,
            "probability": scar_probability,
            "backbone_features": features
        }

    def get_elasticity_map(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output["elasticity"]

    def get_segmentation_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            prob_map = output["probability"]
        return (prob_map > threshold).float()


class ScarElasticReviewerVariant(ScarElasticNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(x)

        e_map = outputs["elasticity"]
        p_map = outputs["probability"]

        alpha = 0.6
        combined_field = alpha * p_map + (1 - alpha) * e_map
        outputs["combined_field"] = combined_field

        return outputs