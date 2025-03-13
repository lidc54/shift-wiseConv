# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# pw：patch-wise


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np
from ops import AddShift_ops 
use_sync_bn = False


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
    # return nn.BatchNorm2d(channels)


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

class lora_module(nn.Module):
    def __init__(self, pad_lk, shift_pads, small_kernel, c_idxes, c_out, c_in) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out=c_out
        self.pad_lk=pad_lk
        self.extra_pad = pad_lk - small_kernel // 2
        self.c1_idxes=torch.arange(self.c_in).int()
        self.c2_idxes=c_idxes.int()
        self.shift_pads=torch.IntTensor(shift_pads)
        self.device = None
     
    def set_attr(self, device, b, hout, wout):
        if self.device is None:
            self.b=b
            self.hout=hout
            self.wout=wout
            self.hin, self.win = hout + self.pad_lk, wout + self.pad_lk
            self.device=device
            self.y1 = torch.zeros(b,self.c_out, self.hout, self.wout, device=device)
            self.y2 = torch.zeros(b,self.c_out, self.hout, self.wout, device=device)
            self.c1_idxes = self.c1_idxes.to(self.device)
            self.c2_idxes = self.c2_idxes.to(self.device)
            self.shift_pads = self.shift_pads.to(self.device)
       
        
    def forward(self, x):
        out = [
            AddShift_ops(x,self.y1,self.c1_idxes,self.shift_pads,self.extra_pad,
                           self.b, self.c_in, self.hin, self.win, 
                           self.c_out, self.hout, self.wout,1),
            AddShift_ops(x,self.y2,self.c2_idxes,self.shift_pads,self.extra_pad,
                           self.b, self.c_in, self.hin, self.win, 
                           self.c_out, self.hout, self.wout,0)
        ]
        return out
        
    # def forward(self, x, b, hout, wout):
    #     if self.device is None:
    #         self.b=b
    #         self.hout=hout
    #         self.wout=wout
    #         self.hin, self.win = hout + self.pad_lk, wout + self.pad_lk
    #         device = x.device
    #         self.device=device
    #         self.y1 = torch.zeros(b,self.c_out, self.hout, self.wout, device=device)
    #         self.y2 = torch.zeros(b,self.c_out, self.hout, self.wout, device=device)
    #         self.c1_idxes = self.c1_idxes.to(self.device)
    #         self.c2_idxes = self.c2_idxes.to(self.device)
    #         self.shift_pads = self.shift_pads.to(self.device)
    #     
    #     out = [
    #         AddShift_ops(x,self.y1,self.c1_idxes,self.shift_pads,self.extra_pad,
    #                        self.b, self.c_in, self.hin, self.win, 
    #                        self.c_out, self.hout, self.wout,1),
    #         AddShift_ops(x,self.y2,self.c2_idxes,self.shift_pads,self.extra_pad,
    #                        self.b, self.c_in, self.hin, self.win, 
    #                        self.c_out, self.hout, self.wout,0)
    #     ]
    #     return out

# shift by padding of torch
class LoRAConvsByRandom_cu(nn.Module):
    '''
    merge LoRA1 LoRA2 small_conv
    random set index for three branch
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
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        # self.split_convs = nn.Conv2d(in_channels, out_n,
        #                              kernel_size=small_kernel, stride=stride,
        #                              padding=padding, groups=group,
        #                              bias=False)
        # only part of input will using shift-wise
        ghost_ratio=0.3
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
        # self.ghost=torch.arange(ghostN).int()
        # self.rep=torch.arange(ghostN, in_channels).int()

        N_rep=4
        out_n = repN * self.nk
        self.split_convs = nn.ModuleList([
            nn.Conv2d(repN, out_n,
                      kernel_size=small_kernel, stride=stride,
                      padding=padding, groups=repN,
                      bias=False)
            for _ in range(N_rep)
        ])
        self.device = None
        self.lora1 = None
        print(f'ghost_ratio={ghost_ratio},N_rep={N_rep}')
        torch.manual_seed(123)
        self.lora2 = torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(repN)])# shuffle in group
        # self.lora2 = torch.cat([torch.arange(self.nk-1,-1,-1) + i * self.nk for i in range(repN)])# shuffle in group
        # np.random.seed(123) 
        # self.small = torch.IntTensor(np.random.randint(0,out_n,out_channels)) if use_small_conv else None
        torch.manual_seed(123)
        self.small = torch.randint(0,self.nk,[repN])+torch.arange(repN)*self.nk if use_small_conv else None
        self.use_bn = bn


        # self.loras=[torch.jit.script(lora_module(padding, real_pad, small_kernel,self.lora2, repN, out_n))
        self.loras=[lora_module(padding, real_pad, small_kernel,self.lora2, repN, out_n)
                    for _ in range(N_rep)]
        if bn:
            self.bn_lora1 = get_bn(repN)
            self.bn_lora2 = get_bn(repN)
            self.bn_lora = get_bn(repN)
            # self.bn_small = get_bn(out_channels) if use_small_conv else None
            # self.bn_lora1 = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
            # self.bn_lora2 = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_lora = None
            # self.bn_lora1 = [None]#*N_rep
            # self.bn_lora2 = [None]*N_rep
        if bn and use_small_conv:
            self.bn_small = get_bn(repN)
            # self.bn_small = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
        else:
            self.bn_small = None
            # self.bn_small = [None]*N_rep
        self.inputsize=True

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
                self.lora2 = self.lora2.to(self.device)
                self.small = self.small.to(self.device) if self.small_conv else None
                self.ghost=self.ghost.to(self.device)
                self.rep=self.rep.to(self.device)
        ghost_inputs = torch.index_select(inputs, 1, self.ghost)
        rep_inputs = torch.index_select(inputs, 1, self.rep)
        lora1_x = 0
        lora2_x = 0
        small_x = 0
        # for (split_convs, bn_lora1, bn_lora2, bn_small) in zip(self.split_convs, self.bn_lora1, self.bn_lora2, self.bn_small):
        # outs=torch.load('se.pt')
        for split_convs, lora in zip(self.split_convs, self.loras):
        # for out, lora in zip(outs, self.loras):
            out = split_convs(rep_inputs)
            lora.set_attr(out.device, ori_b, ori_h, ori_w)
            x1,x2=lora(out)
            # def forward(self, x, b, hin, win):
            lora1_x += x1 #self.forward_lora(out, ori_h, ori_w)#, bn=bn_lora1)
            lora2_x += x2 #self.forward_lora(out, ori_h, ori_w, idx=self.lora2, VH='W')#, bn=bn_lora2)
            if self.small_conv:
                small_x += self.forward_small(out, self.small)#, bn_small
        if self.use_bn:
            lora1_x = self.bn_lora1(lora1_x)
            lora2_x = self.bn_lora2(lora2_x) 
            small_x = self.bn_small(small_x) 
        x = lora1_x + lora2_x + small_x
        if self.use_bn:
            x = self.bn_lora(x) 
        x=torch.cat([x,ghost_inputs],dim=1)
        return x
        #     # out_rep = self.split_rep_convs(inputs)
        #         # small_rep_x = self.forward_small(out_rep, self.small, self.bn_rep_small)
        #         # x_rep += small_rep_x
        # # y=x+x_rep
        #     # x_rep = lora1_rep_x + lora2_rep_x
        #     # lora1_rep_x = self.forward_lora(out_rep, ori_h, ori_w, bn=self.bn_rep_lora1)
        #     # lora2_rep_x = self.forward_lora(out_rep, ori_h, ori_w, VH='W', idx=self.lora2, bn=self.bn_rep_lora2)

    def forward_small(self, out, idx): #, small_bn):
        # shift along the index of every group
        b, c, h, w = out.shape
        out = torch.index_select(out, 1, idx)
        padding, *_ = self.pad
        k = min(self.kernels) 
        pad = padding - k // 2
        if pad>0:
            out = torch.narrow(out, 2, pad, h - 2 * pad)
            out = torch.narrow(out, 3, pad, w - 2 * pad)
        # if self.use_bn:
        #     out = small_bn(out)
        return out

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
        # torch.jit.script(
        return LoRAConvsByRandom_cu(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,
                                 group=groups, stride=stride,
                                 use_small_conv=use_small_conv)#)
        # return LoRAConvsByRandomTaichi(in_channels, out_channels, bn=bn,
        #                          big_kernel=big_kernel, small_kernel=small_kernel,
        #                          group=groups, stride=stride,
        #                          use_small_conv=use_small_conv)


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
            # todo:
            use_small_conv = (small_kernel is not None) and small_kernel < kernel_size
            self.LoRA = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, small_kernel),
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bn=bn,
                use_small_conv=use_small_conv
            )
            # if self.Decom:
            #     self.LoRA1 = conv_bn(
            #         in_channels=in_channels,
            #         out_channels=out_channels,
            #         kernel_size=(kernel_size, small_kernel),
            #         stride=stride,
            #         padding=padding,
            #         dilation=1,
            #         groups=groups,
            #         bn=bn,
            #     )
            #     self.LoRA2 = conv_bn(
            #         in_channels=in_channels,
            #         out_channels=out_channels,
            #         kernel_size=(small_kernel, kernel_size),
            #         stride=stride,
            #         padding=padding,
            #         dilation=1,
            #         groups=groups,
            #         bn=bn,
            #     )
            # else:
            #     self.lkb_origin = conv_bn(
            #         in_channels=in_channels,
            #         out_channels=out_channels,
            #         kernel_size=kernel_size,
            #         stride=stride,
            #         padding=padding,
            #         dilation=1,
            #         groups=groups,
            #         bn=bn,
            #     )
            #
            # if (small_kernel is not None) and small_kernel < kernel_size:
            #     self.small_conv = conv_bn(
            #         in_channels=in_channels,
            #         out_channels=out_channels,
            #         kernel_size=small_kernel,
            #         stride=stride,
            #         padding=small_kernel // 2,
            #         groups=groups,
            #         dilation=1,
            #         bn=bn,
            #     )

    def forward(self, inputs):
        out = self.LoRA(inputs)
        # if hasattr(self, "small_conv"):
        #     out += self.small_conv(inputs)

        # if hasattr(self, "lkb_reparam"):
        #     out = self.lkb_reparam(inputs)
        # elif self.Decom:
        #     out = self.LoRA1(inputs) + self.LoRA2(inputs)
        #     if hasattr(self, "small_conv"):
        #         out += self.small_conv(inputs)
        # else:
        #     out = self.lkb_origin(inputs)
        #     if hasattr(self, "small_conv"):
        #         out += self.small_conv(inputs)
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
    r"""SLaK_gr_swsf_se_cu Block. There are two equivalent implementations:
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

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.se = nn.Sequential(
            SEBlock(dim, dim // 4),
            NCHWtoNHWC()
        )

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
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # x = self.norm(x)
        x = self.se(self.norm(x))
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# patch-wise
class SLaK_gr_swsf_se_cu(nn.Module):
    r"""SLaK_gr_swsf_se_cu
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
# fp:fixing padding
@register_model
def SLaK_gr_swsf_se_cu_tiny(pretrained=False, **kwargs):
    model = SLaK_gr_swsf_se_cu(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def SLaK_gr_swsf_se_cu_small(pretrained=False, **kwargs):
    model = SLaK_gr_swsf_se_cu(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def SLaK_gr_swsf_se_cu_base(pretrained=False, in_22k=False, **kwargs):
    model = SLaK_gr_swsf_se_cu(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

    return model


@register_model
def SLaK_gr_swsf_se_cu_large(pretrained=False, in_22k=False, **kwargs):
    model = SLaK_gr_swsf_se_cu(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

    return model

