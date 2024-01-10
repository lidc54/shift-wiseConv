import sys, os
# Add WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension into your PYTHONPATH by the following commands:
sys.path.append('SLaK/cutlass/examples/19_large_depthwise_conv2d_torch_extension')
fp = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, fp)
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
from models.SLaK_reg import ConvGroupShift
from utils import reloc_w
import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension
import time,math



# 4. mini kernel as main character
class ConvGroupTest(nn.Module):
    #只有conv没有shift的速度，查看速度的baseline
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel, stride=1, group=1,
                 bn=True, weight=None):
        super().__init__()
        assert len(set(kernel)) == 2, "must have two different kernel size"
        mink, maxk = min(kernel), max(kernel)
        self.kernels = kernel
        self.stride = stride
        if (mink, maxk) == kernel:
            self.VH = 'H'  # 横向
            padding=(mink // 2, mink-1)
        else:
            self.VH = 'V'
            padding=(mink-1, mink // 2)
        # padding, after_padding_index, index = self.shift(kernel)
        # padding = (padding, mink // 2) if self.VH == 'V' else (mink // 2, padding)
        # self.pad = after_padding_index, index
        # print(padding, after_padding_index, index)
        self.nk = math.ceil(maxk / mink)
        self.split_convs = nn.Conv2d(in_channels, out_channels * self.nk,
                                     kernel_size=mink,  stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)
        # self.reloc_weight(weight)
        # self.use_bn = bn
        # if bn:
        #     self.bn = get_bn(out_channels)

    def forward(self, inputs):
        out = self.split_convs(inputs)
        # b, c, h, w = out.shape
        # # split output
        # *_, ori_h, ori_w = inputs.shape
        # # out = torch.split(out, c // self.nk, 1)
        # out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        # x = 0
        # for i in range(self.nk):
        #     outi = self.rearrange_data(out[i], i, ori_h, ori_w)
        #     x = x + outi
        # if self.use_bn:
        #     x = self.bn(x)
        # return x
            
if __name__ == "__main__":
    torch.random.manual_seed(0)
    X=51
    if torch.cuda.is_available():
        x = torch.randn(64, 384, X+5, X+5).cuda()
        m1 = DepthWiseConv2dImplicitGEMM(384, (X,5), bias=False).cuda()
        m2 = nn.Conv2d(384, 384, (X,5), padding=(X // 2,2), bias=False, groups=384).cuda()
        mReg=ConvGroupShift(384,384,(X,5),group=384,bn=False).cuda()
        mTst=ConvGroupTest(384,384,(X,5),group=384,bn=False).cuda()
        m2.load_state_dict(m1.state_dict())
        new_dict={}
        for k,k1 in zip(m1.state_dict(), mReg.state_dict()):
            new_dict[k1]=reloc_w(m1.state_dict()[k])
        mReg.load_state_dict(new_dict)
        with torch.cuda.amp.autocast(True):
            t0=time.time()
            y1 = m1(x)
            t1=time.time()
            y2 = m2(x)
            t2=time.time()
            y3 = mReg(x)
            t3=time.time()
            y = mTst(x)
            t4=time.time()
        (y1.mean() * 1024).backward()
        (y2.mean() * 1024).backward()
        (y3.mean() * 1024).backward()
        print("y1-y2 output difference:", ((y1 - y2) ** 2).mean())
        print("y1-y2 gradient difference:", ((m1.weight.grad - m2.weight.grad) ** 2).mean())
        print("y3-y2 output difference:", ((y3 - y2) ** 2).mean())
        print("y3-y2 gradient difference:", (mReg.split_convs.weight.grad.mean() - m2.weight.grad.mean()))
        print(f't1={t1-t0}s, t2={t2-t1}s, t3={t3-t2}s, t4={t4-t3}s')
