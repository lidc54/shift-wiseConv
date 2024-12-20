# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# pw：patch-wise with random generate connection


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np

use_sync_bn = True


# 2. depth wise convolution
def get_conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
):
    # return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    try:
        paddings = (kernel_size[0] // 2, kernel_size[1] // 2)
    except Exception as e:
        paddings = padding
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, paddings, dilation, groups, bias
    )


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn_ori(
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True
):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
    )

    if bn:
        result.add_module("bn", get_bn(out_channels))
    return result


class Mask(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(*size), requires_grad=True)
        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        w = torch.sigmoid(self.weight)
        masked_wt = w.mul(x)
        return masked_wt


class LoRAConvsByWeight(nn.Module):
    '''
    merge LoRA1 LoRA2
    shuffle channel by weights rather index
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 big_kernel, small_kernel,
                 stride=1, group=1,
                 bn=True, use_small_conv=True):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        self.small_conv = use_small_conv
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, after_padding_index, index = self.shift(self.kernels)
        self.pad = padding, after_padding_index, index
        self.nk = math.ceil(big_kernel / small_kernel)
        out_n = out_channels * self.nk
        self.split_convs = nn.Conv2d(in_channels, out_n,
                                     kernel_size=small_kernel, stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)

        self.lora1 = Mask((1, out_n, 1, 1))
        self.lora2 = Mask((1, out_n, 1, 1))
        self.use_bn = bn

        if bn:
            self.bn_lora1 = get_bn(out_channels)
            self.bn_lora2 = get_bn(out_channels)
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None

    def forward(self, inputs):
        out = self.split_convs(inputs)
        # split output
        *_, ori_h, ori_w = inputs.shape
        lora1_x = self.forward_lora(self.lora1(out), ori_h, ori_w, VH='H', bn=self.bn_lora1)
        lora2_x = self.forward_lora(self.lora2(out), ori_h, ori_w, VH='W', bn=self.bn_lora2)
        x = lora1_x + lora2_x
        return x

    def forward_lora(self, out, ori_h, ori_w, VH='H', bn=None):
        # shift along the index of every group
        b, c, h, w = out.shape
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        x = 0
        for i in range(self.nk):
            outi = self.rearrange_data(out[i], i, ori_h, ori_w, VH)
            x = x + outi
        if self.use_bn:
            x = bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w, VH):
        padding, _, index = self.pad
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        # need to calculate start point after conv
        # how many windows shift from real start window index
        if (idx + 1) >= index:
            pad_l = 0
            s = (idx + 1 - index) * (k // stride)
        else:
            pad_l = (index - 1 - idx) * (k // stride)
            s = 0
        if VH == 'H':
            # assume add sufficient padding for origin conv
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            # e = w + pad_l + pad_r - s - suppose_len
        else:
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            # e = h + pad_l + pad_r - s - suppose_len
        # print('new_pad', new_pad)
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
        # split_list = [s, suppose_len, e]
        # padding on v direction
        if padding * 2 + 1 != k:
            pad = padding - k // 2
            if VH == 'H':  # horizonal
                x = torch.narrow(x, 2, pad, h - 2 * pad)
            else:  # vertical
                x = torch.narrow(x, 3, pad, w - 2 * pad)

        xs = torch.narrow(x, dim, s, suppose_len)
        return xs

    def shift(self, kernels):
        '''
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.
        '''
        mink, maxk = min(kernels), max(kernels)
        mid_p = maxk // 2
        # 1. new window size is mink. middle point index in the window
        offset_idx_left = mid_p % mink
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink
        # 2. padding
        padding = offset_idx_left % mink
        while padding < offset_idx_right:
            padding += mink
        # 3. make sure last pixel can be scan by min window
        while padding < (mink - 1):
            padding += mink
        # 4. index of windows start point of middle point
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink
        # 5. output:padding how to padding input in v&h direction;
        # after_padding_index: middle point of original kernel will located in which window
        # real_start_idx: start window index after padding in original kernel along long side
        return padding, after_padding_index, real_start_idx


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True, use_small_conv=True):
    if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:
        return conv_bn_ori(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bn)
    else:
        big_kernel, small_kernel = kernel_size
        return LoRAConvsByWeight(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,
                                 group=groups, stride=stride,
                                 use_small_conv=use_small_conv)


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


class ReparamLargeKernelConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            small_kernel,
            small_kernel_merged=False,
            Decom=False,
            bn=True,
    ):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:  # cpp版本的conv，加快速度
            self.lkb_reparam = get_conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            if self.Decom:
                self.LoRA = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, small_kernel),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn
                )
                # self.LoRA1 = conv_bn(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=(kernel_size, small_kernel),
                #     stride=stride,
                #     padding=padding,
                #     dilation=1,
                #     groups=groups,
                #     bn=bn,
                # )
                # self.LoRA2 = conv_bn(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=(small_kernel, kernel_size),
                #     stride=stride,
                #     padding=padding,
                #     dilation=1,
                #     groups=groups,
                #     bn=bn,
                # )
            else:
                self.lkb_origin = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )

            if (small_kernel is not None) and small_kernel < kernel_size:
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bn=bn,
                )

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            # out = self.LoRA1(inputs) + self.LoRA2(inputs)
            out = self.LoRA(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


class Block(nn.Module):
    r"""SLaK_pww Block. There are two equivalent implementations:
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
            Decom=None,
            bn=True,
    ):
        super().__init__()

        self.large_kernel = ReparamLargeKernelConv(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size[0],
            stride=1,
            groups=dim,
            small_kernel=kernel_size[1],
            small_kernel_merged=False,
            Decom=Decom,
            bn=bn,
        )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.large_kernel(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# patch-wise
class SLaK_pww(nn.Module):
    r"""SLaK_pww
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
            Decom=None,
            bn=True,
            **kwargs,
    ):
        super().__init__()
        dims = [int(x * width_factor) for x in dims]
        self.kernel_size = kernel_size
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
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
                        Decom=Decom,
                        bn=bn,
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


# pw：patch-wise
# w: with weight
@register_model
def SLaK_pww_tiny(pretrained=False, **kwargs):
    model = SLaK_pww(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def SLaK_pww_small(pretrained=False, **kwargs):
    model = SLaK_pww(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def SLaK_pww_base(pretrained=False, in_22k=False, **kwargs):
    model = SLaK_pww(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

    return model


@register_model
def SLaK_pww_large(pretrained=False, in_22k=False, **kwargs):
    model = SLaK_pww(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

    return model

