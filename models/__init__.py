import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from .scarelastic_net import ScarElasticNet
from .backbone import UNet3DEncoder, UNet3DDecoder

logger = logging.getLogger(__name__)

__all__ = [
    "ScarElasticNet",
    "UNet3DEncoder",
    "UNet3DDecoder",
    "ModelFactory",
    "init_weights",
    "count_parameters",
    "load_pretrained_weights"
]


def init_weights(net: nn.Module, init_type: str = 'kaiming', gain: float = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented.')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm3d') != -1 or classname.find('InstanceNorm3d') != -1:
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def count_parameters(model: nn.Module, verbose: bool = True) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")

    return trainable_params


def load_pretrained_weights(model: nn.Module, path: str, strict: bool = True, map_location: str = 'cpu'):
    if not path:
        return model

    try:
        state_dict = torch.load(path, map_location=map_location)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(pretrained_dict.keys())

        if len(missing_keys) > 0:
            logger.warning(f"Missing keys in pretrained model: {list(missing_keys)[:5]} ...")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys in pretrained model: {list(unexpected_keys)[:5]} ...")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=strict)
        logger.info(f"Successfully loaded pretrained weights from {path}")

    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        raise


class ModelFactory:
    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        model_name = config.get("name", "scarelastic").lower()

        if model_name == "scarelastic":
            model = ScarElasticNet(
                in_channels=config.get("in_channels", 4),
                out_channels=config.get("out_channels", 1),
                base_filters=config.get("base_filters", 16),
                depth=config.get("depth", 4),
                use_se=config.get("use_se", True),
                spatial_dims=3,
                dropout_rate=config.get("dropout", 0.0)
            )
        elif model_name == "unet3d_baseline":
            from monai.networks.nets import UNet
            model = UNet(
                spatial_dims=3,
                in_channels=config.get("in_channels", 4),
                out_channels=config.get("out_channels", 1),
                channels=[16, 32, 64, 128, 256],
                strides=[2, 2, 2, 2],
                num_res_units=2,
                norm="instance"
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

        if config.get("init_weights", True):
            init_weights(model, init_type="kaiming")

        if config.get("pretrained_path"):
            load_pretrained_weights(model, config["pretrained_path"])

        return model

    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: list):
        for name, param in model.named_parameters():
            for target in layer_names:
                if target in name:
                    param.requires_grad = False
                    logger.info(f"Freezing layer: {name}")

    @staticmethod
    def unfreeze_all(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True