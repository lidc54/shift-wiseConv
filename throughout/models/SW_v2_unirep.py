# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# pw：patch-wise


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from shiftadd.ops import AddShift_mp_blur_module, AddShift_mp_module
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath

use_sync_bn = True
use_se = True

class SwiGLU(nn.Module):
    # https://zhuanlan.zhihu.com/p/650237644
    def __init__(self, dim: int, hidden_dim: int, dropout: float): #multiple_of: int,
        super().__init__()
        # hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
     
class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
    
def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

class SEBlock(nn.Module):
    """
    https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py#L120C1-L141C62
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ShifthWiseConv2dImplicit(nn.Module):
    '''
    will be displayed soon
    '''
    pass
 
def get_conv2d(in_channels, out_channels, big_kernel, small_kernel=3, stride=1, group=1, bn=True, use_small_conv=True, ghost_ratio=0.23, N_path=2, N_rep=4, bias=False, use_lk_impl=True):
    if use_lk_impl:
        return ShifthWiseConv2dImplicit(
            in_channels=in_channels,
            out_channels=out_channels,
            big_kernel=big_kernel,
            small_kernel=small_kernel,
            stride=stride,
            group=group,
            bn=bn,
            ghost_ratio=ghost_ratio,
            N_path=N_path, # multi path
            N_rep=N_rep
        )
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel, stride=stride,
                     padding=small_kernel//2, groups=group, bias=bias)

class Block(nn.Module):
    r"""ShiftWise_v2 Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            drop_path=0.0,
            layer_scale_init_value=1e-6,
            kernel_size=(7, 7),
            ghost_ratio=0.23,
            N_path=2, # multi path
            N_rep=4, # multi split
            bn=True,
            deploy=False,
            use_lk_impl=True
    ):
        super().__init__()

        self.large_kernel = get_conv2d(in_channels=dim, out_channels=dim, big_kernel=kernel_size[0], small_kernel=kernel_size[1], stride=1, group=dim, bn=bn, ghost_ratio=ghost_ratio, N_path=N_path, N_rep=N_rep, use_lk_impl=use_lk_impl)

        if use_se:
            self.norm = get_bn(dim)
            self.se = SEBlock(dim, dim // 4)
            self.pwconv1 = nn.Sequential(
                NCHWtoNHWC(),
                nn.Linear(dim, 4 * dim)
                )
        else:
            self.norm = LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, 4 * dim)

        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(4 * dim, use_bias=not deploy)
            )
        
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(4 * dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(4 * dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim))
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if (not deploy) and layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.large_kernel(x)

        if use_se:
            x = self.se(self.norm(x))
        else:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)

        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input + self.drop_path(x)
        return x

    def reparameterize(self):
        if hasattr(self.large_kernel, 'merge_branches'):
            self.large_kernel.merge_branches()

# patch-wise
class ShiftWise_v2(nn.Module):
    r"""ShiftWise_v2
        A PyTorch impl of More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            head_init_scale=1.0,
            kernel_size=[51, 49, 45, 13, 5],
            width_factor=1.0,
            bn=True,
            ghost_ratio=0.23,
            N_path=2, # multi path
            N_rep=4, # multi split
            left_first_stage=False,
            deploy=False,
            **kwargs,
    ):
        super().__init__()
        # print(f'dims:{dims}, depths:{depths}')
        dims = [int(x * width_factor) for x in dims]
        self.kernel_size = kernel_size
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            # nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                # nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=(self.kernel_size[i], self.kernel_size[-1]),
                        ghost_ratio=ghost_ratio,
                        N_path=N_path, # multi path
                        N_rep=N_rep, # multi split
                        bn=bn,
                        deploy=deploy,
                        use_lk_impl=True
                        # use_lk_impl=True if not left_first_stage else  (i>0) # use general dw for first stage
                        # use_lk_impl= (i>0) # use general dw for first stage
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            # print(m.bias)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def ShiftWise_v2_tiny(pretrained=False, **kwargs):
    model = ShiftWise_v2(depths=[3, 3, 18, 3], dims=[80, 160, 320, 640], **kwargs)

    return model


@register_model
def ShiftWise_v2_small(pretrained=False, **kwargs):
    model = ShiftWise_v2(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    return model
