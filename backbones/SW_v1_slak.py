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
from shiftadd.ops import AddShift_mp_module
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath

use_sync_bn = True
use_se = False


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
    Using shift is equivalent to using a large convolution kernel.
    '''
    def __init__(self,
                in_channels,
                out_channels,
                big_kernel,
                small_kernel=3,
                stride=1,
                group=1,
                bn=True,
                use_small_conv=True,
                ghost_ratio=0.23,
                N_path=2, # multi path
                N_rep=4, # multi split
                bias=False):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        
        # only part of input will using shift-wise
        ghostN=int(in_channels*ghost_ratio)
        repN= in_channels - ghostN
        np.random.seed(123)  
        ghost = np.random.choice(in_channels,ghostN,replace=False).tolist()
        ghost.sort()
        rep=list(set(range(in_channels))-set(ghost))
        rep.sort()
        assert len(rep)==repN,f'len(rep):{len(rep)}==repN:{repN}'
        self.ghost=torch.IntTensor(ghost)
        self.rep=torch.IntTensor(rep)
        # self.register_buffer('rep1', torch.IntTensor(rep))
        # self.register_buffer('ghost', torch.IntTensor(ghost))

        out_n = repN * self.nk
        self.LoRA = None
        self.LoRAs = nn.ModuleList([
            nn.Conv2d(repN, out_n,
                kernel_size=small_kernel, stride=stride,
                padding=padding, groups=repN,
                bias=False)
            for _ in range(N_path)
        ])
        self.device = None
        print(f'ghost_ratio={ghost_ratio},N_rep={N_rep},N_path={N_path}')
        self.use_bn = bn

        self.loras=AddShift_mp_module(big_kernel, small_kernel, repN, out_n, N_rep)
        self.bn_loras = None
        # self.bn_sum = get_bn(repN)
        # print("use bn layer after multi path shift add")
        if bn:
            # self.bn_loras = nn.ModuleList([get_bn(out_n) for _ in range(N_path)])
            self.bn_lora1 = get_bn(repN)
            self.bn_lora2 = get_bn(repN)
            self.bn_small = get_bn(repN)
        else:
            self.bn_loras = None
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_small = None

        self.inputsize=True

    def ghost_mask(self):
        # all output is masked 
        weight=0
        for name, tensor in self.named_parameters():
            if len(tensor.size()) == 4:
                weight += torch.sum(torch.abs(tensor.detach()),(1,2,3))
        weight = torch.sum((weight>0).reshape(-1, self.nk), 1)
        ghost = (weight==0).reshape(1,-1,1,1).float() 
        return ghost
    
    def forward(self, inputs):
        if self.inputsize:
            print(f'shape:{inputs.shape}; kernel:{self.kernels}')
            self.inputsize=False
            
        # split output
        ori_b, ori_c, ori_h, ori_w = inputs.shape
        if self.device is None:
            self.device = inputs.get_device()
            if self.device ==-1:#cpu
                self.device=None
            else:
                self.ghost=self.ghost.to(self.device)
                self.rep=self.rep.to(self.device)
        
        ghost_inputs = torch.index_select(inputs, 1, self.ghost)
        rep_inputs = torch.index_select(inputs, 1, self.rep)
        
        lora1_x = 0
        lora2_x = 0
        small_x = 0
        #old
        # for split_conv, lora in zip(self.LoRAs, self.loras):
        #     out = split_conv(rep_inputs)
        #     x1, x2, x3 = lora(out, ori_b, ori_h, ori_w)
        #     lora1_x += x1
        #     lora2_x += x2
        #     small_x += x3
        #new
        out=0
        
        if self.LoRA is None: 
            for ii, split_conv in enumerate(self.LoRAs):
                xx = split_conv(rep_inputs)
                # if self.bn_loras is not None:
                #     xx = self.bn_loras[ii](xx)
                out += xx
        else:
            out=self.LoRA(rep_inputs)

        x1, x2, x3 = self.loras(out, ori_b, ori_h, ori_w)
        lora1_x += x1
        lora2_x += x2
        small_x += x3

        if self.use_bn:
            lora1_x = self.bn_lora1(lora1_x)
            lora2_x = self.bn_lora2(lora2_x) 
            small_x = self.bn_small(small_x)
        x = lora1_x + lora2_x + small_x + rep_inputs
        # x = self.bn_sum(x)
        # g_mask = self.ghost_mask()
        # x = lora1_x + lora2_x + small_x + rep_inputs * g_mask
 
        x=torch.cat([x,ghost_inputs],dim=1)
        return x
     

    def shift(self, kernels):
        '''
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.
        '''
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink) 
        # 2. padding
        padding = mink -1  
        # padding = mink // 2
        # 3. pads for each idx
        mid=maxk // 2
        real_pad=[]
        for i in range(nk): 
            extra_pad=mid-i*mink - padding 
            real_pad.append(extra_pad)
        return padding, real_pad

    def merge_branches(self):
        if self.LoRA is None: 
            bias=True if self.LoRAs[0].bias else False
            LoRA = nn.Conv2d(in_channels=self.LoRAs[0].in_channels,
                            out_channels=self.LoRAs[0].out_channels,
                            kernel_size=self.LoRAs[0].kernel_size,
                            stride=self.LoRAs[0].stride,
                            padding=self.LoRAs[0].padding,
                            dilation=self.LoRAs[0].dilation,
                            groups=self.LoRAs[0].groups,
                            bias=bias
                            )
            weight,biasdata=0,0
            for merged_conv in self.LoRAs:
                weight+=merged_conv.weight.data
                if bias:
                    biasdata+=merged_conv.bias.data
            
            LoRA.weight.data = weight
            if bias: LoRA.bias.data = biasdata
            self.LoRA=LoRA
            self.__delattr__('LoRAs') 

class Block(nn.Module):
    r"""ShiftWise_v1 Block. There are two equivalent implementations:
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
    ):
        super().__init__()

        self.large_kernel = ShifthWiseConv2dImplicit(
            in_channels=dim,
            out_channels=dim,
            big_kernel=kernel_size[0],
            small_kernel=kernel_size[1],
            stride=1,
            group=dim,
            bn=bn,
            # small_kernel_merged=False,
            # use_small_conv=True,
            ghost_ratio=ghost_ratio,
            N_path=N_path, # multi path
            N_rep=N_rep, # multi split
            # bias=False
        )

        if use_se:
            self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
            self.se = nn.Sequential(
                SEBlock(dim, dim // 4),
                NCHWtoNHWC()
            )
        else:
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
        if use_se:
            x = self.se(self.norm(x))
        else:
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

    def reparameterize(self):
        if hasattr(self.large_kernel, 'merge_branches'):
            self.large_kernel.merge_branches()
        # if self.gamma is not None:
        #     final_scale = self.gamma.data
        #     self.gamma = None
        # else:
        #     final_scale = 1
        # if self.act[1].use_bias and len(self.pwconv2) == 3:
        #     grn_bias = self.act[1].beta.data
        #     self.act[1].__delattr__('beta')
        #     self.act[1].use_bias = False
        #     linear = self.pwconv2[0]
        #     grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
        #     bn = self.pwconv2[2]
        #     std = (bn.running_var + bn.eps).sqrt()
        #     new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        #     new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
        #     linear_bias = 0 if linear.bias is None else linear.bias.data
        #     linear_bias += grn_bias_projected_bias
        #     new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
        #     self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])

# patch-wise
class ShiftWise_v1(nn.Module):
    r"""ShiftWise_v1
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
                        ghost_ratio=ghost_ratio,
                        N_path=N_path, # multi path
                        N_rep=N_rep, # multi split
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


@register_model
def ShiftWise_v1_tiny(pretrained=False, **kwargs):
    model = ShiftWise_v1(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def ShiftWise_v1_small(pretrained=False, **kwargs):
    model = ShiftWise_v1(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def ShiftWise_v1_base(pretrained=False, in_22k=False, **kwargs):
    model = ShiftWise_v1(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

    return model


@register_model
def ShiftWise_v1_large(pretrained=False, in_22k=False, **kwargs):
    model = ShiftWise_v1(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

    return model
