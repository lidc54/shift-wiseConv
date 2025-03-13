import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np
from torch.autograd import Variable
from ops import AddShift_ops

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000:.4f} ms")


# shift by padding of torch
class LoRAConvsByRandom(nn.Module):
    '''
    merge LoRA1 LoRA2 small_conv
    random set index for three branch
    '''

    def __init__(self, big_kernel, small_kernel, stride=1):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride 
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
    
    def forward(self, inputs, ori_h, ori_w,isH):
        if isH:
            VH='H'
        else:
            VH='V'
        x = self.forward_lora(inputs, ori_h, ori_w,VH)#, bn=bn_lora1)
        return x
    
    def forward_lora(self, out, ori_h, ori_w, VH='H', idx=None, bn=None):
        # shift along the index of every group
        b, c, h, w = out.shape
        if idx is not None:
            # out=out[idx]
            out = torch.index_select(out, 1, idx)
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        x = 0
        for i in range(self.nk):
            outi = self.rearrange_data(out[i], i, ori_h, ori_w, VH)
            x = x + outi
        # if self.use_bn:
        #     x = bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w, VH):
        padding, pads = self.pad
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        pad=pads[idx] 

        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        # need to calculate start point after conv
        # how many windows shift from real start window index
        if pad<0:
            pad_l = 0
            s = 0-pad
        else:
            pad_l = pad
            s = 0
        if VH == 'H':
            # assume add sufficient padding for origin conv
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            e = w + pad_l + pad_r - s - suppose_len
        else:
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            e = h + pad_l + pad_r - s - suppose_len
        # print('new_pad', new_pad)
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
         
        # padding on other direction
        # if padding * 2 + 1 != k:
        #     pad = padding - k // 2
        yy = padding - k // 2
        if yy>0:
            if VH == 'H':  # horizonal
                # x = torch.narrow(x, 2, pad, h - 2 * pad)
                x = torch.narrow(x, 2, yy, h - 2 * yy)
            else:  # vertical
                # x = torch.narrow(x, 3, pad, w - 2 * pad)
                x = torch.narrow(x, 3, yy, w - 2 * yy)

        xs = torch.narrow(x, dim, s, suppose_len)
        return xs

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

def compare_addshift_LoRAConvsByRandom(big_kernel, small_kernel,b,c,h,w):
    print('**********************\n export CUDA_VISIBLE_DEVICES=7') 
    # big_kernel, small_kernel=51,5
    # b,c,h,w=2,128,61,61
    nk = math.ceil(big_kernel / small_kernel)
    sw = LoRAConvsByRandom(big_kernel, small_kernel)
    padding, pads = sw.pad
    b, c_in, hin, win = b, c*nk,h+padding,w+padding
    c_out, hout, wout = c,h,w 
    h_pad = padding - small_kernel // 2
    idxes=torch.arange(c*nk).int().cuda()
    shift_pads=torch.IntTensor(pads).cuda()

    ipt=np.random.rand(b,c_in, hin, win).astype(np.float32)
    target=np.random.rand(b,c,h,w).astype(np.float32)
    x0 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    x1 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    # x2 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    # y1 = Variable(torch.zeros(b,c,h,w).cuda())
    # y1 = torch.zeros(b,c,h,w).cuda()
    # y2 = Variable(torch.zeros(b,c,h,w).cuda())
    z = Variable(torch.tensor(target).cuda())

    for isH in [0,1]:
        print('\ny0', '--'*10)
        with Timer("LoRAConvsByRandom"):
            y0=sw(x0,h,w,isH) 
            l0 = F.mse_loss(y0, z)
            l0.backward()
            gx0=x0.grad

        ####################################
        
        with Timer("AddShift_ops"):
            ans = AddShift_ops(x1,idxes,shift_pads,h_pad,b, c_in, hin, win, c_out, hout, wout,isH)
            l1 = F.mse_loss(ans, z)
            l1.backward()
            gx1=x1.grad
        print(torch.all(torch.abs(y0-ans)<1e-3))
        print(torch.all(torch.abs(gx0-gx1)<1e-3))
        print('####################################')
        # print(y0.sum().data,y0.mean().data,'******\n', ans.sum().data,ans.mean().data)
        # print(gx0.sum().data,gx0.mean().data,'******\n', gx1.sum().data,gx1.mean().data)
        # yy=torch.abs(y0-ans)
        # print('diff',yy.sum(),yy.mean())
        # print('y1', '--'*10)
            
        # print('####################################')
        # addshift.forward(y2,x0,idxes,shift_pads,h_pad,b, c_in, hin, win, c_out, hout, wout,isH) 
        # torch.cuda.synchronize(device="cuda:0")
        # print(y0.sum(),y0.mean(),'******\n', y2.sum(),y2.mean())
        # print('--'*10)
 
    
if __name__ =="__main__":
    import random
    for _ in range(50):
        bk= random.randint(7,53)//2*2+1
        sk= random.randint(2,7)//2*2+1
        b,c=random.randint(1,9),random.randint(2,30)
        h,w=random.randint(bk+21,bk+40),random.randint(bk+21,bk+40)
        print(bk, sk,b,c,h,w)
        compare_addshift_LoRAConvsByRandom(bk, sk,b,c,h,w)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        
    



