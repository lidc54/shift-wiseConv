import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ops import AddShift_mp_blur_module, AddShift_mp_module, AddShift_ops
import numpy as np

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

class lora_module_clone(nn.Module):
    # clone from models/SLaK_gr2_sw_sf_se_mp.py
    def __init__(self, pad_lk, shift_pads, small_kernel, c_out, c_in) -> None: 
        super().__init__()
        self.c_in = c_in
        self.c_out=c_out
        self.pad_lk=pad_lk
        self.extra_pad = pad_lk - small_kernel // 2
        self.shift_pads=torch.IntTensor(shift_pads)
        self.device = None
     
    def forward(self, x, b, hout, wout, c1_idxes, c2_idxes):
        if self.device is None:
            self.hin, self.win = hout + self.pad_lk, wout + self.pad_lk
            device = x.device
            self.device=device
            self.shift_pads = self.shift_pads.to(self.device)

        out = [AddShift_ops(x,c1_idxes,self.shift_pads,self.extra_pad,
                            b, self.c_in, self.hin, self.win,
                            self.c_out, hout, wout,1),
               AddShift_ops(x,c2_idxes,self.shift_pads,self.extra_pad,
                            b, self.c_in, self.hin, self.win,
                            self.c_out, hout, wout,0)
        ]
        return out

class LoRAConvsByRandom_cu_clone(nn.Module):
    # clone from models/SLaK_gr2_sw_sf_se_mp.py

    def __init__(self,
                 in_channels: int,
                 big_kernel, small_kernel, N_rep):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        
        repN= in_channels
        # N_path=2 # multi path
        # N_rep=1 # multi split
        
        out_n = repN * self.nk
        self.device = None
        self.lora1 = None
        torch.manual_seed(123)
        self.lora1 = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(repN)]).int() for _ in range(N_rep)]# shuffle in group
        self.lora2 = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(repN)]).int() for _ in range(N_rep)]# shuffle in group
        self.small = [(torch.randint(0,self.nk,[repN])+torch.arange(repN)*self.nk).int() for _ in range(N_rep)]
        # print(self.nk,repN,N_rep)
        
        # print('cu_idx_horizon', self.lora1 )
        # print('cu_idx_vertica',self.lora2)
        # print('cu_idx_identit',self.small)
        self.loras=[lora_module_clone(padding, real_pad, small_kernel, repN, out_n) for _ in range(N_rep)]
        #     self.bn_lora1 = get_bn(repN)
        #     self.bn_lora2 = get_bn(repN)
        #     self.bn_small = get_bn(repN)

    def forward(self, inputs, ori_h, ori_w):
        # split output
        ori_b, *_ = inputs.shape
        if self.device is None:
            self.device = inputs.get_device()
            if self.device ==-1:#cpu
                self.device=None
            else:
                self.lora1 = [lr.to(self.device) for lr in self.lora1]
                self.lora2 = [lr.to(self.device) for lr in self.lora2]
                self.small = [sm.to(self.device) for sm in self.small]
        
        lora1_x = 0
        lora2_x = 0
        small_x = 0
        # print(inputs.shape, ori_b, ori_h, ori_w)
        for content in zip(self.loras, self.lora1, self.lora2, self.small):
            lora, lora1, lora2, small = content
            x1,x2=lora(inputs, ori_b, ori_h, ori_w, lora1, lora2)
            x3 = self.forward_small(inputs, small)
            lora1_x += x1
            lora2_x += x2
            small_x += x3
        # if self.use_bn:
        #     lora1_x = self.bn_lora1(lora1_x)
        #     lora2_x = self.bn_lora2(lora2_x) 
        #     small_x = self.bn_small(small_x)

        return lora1_x, lora2_x, small_x
       
    def forward_small(self, out, idx):
        # shift along the index of every group
        b, c, h, w = out.shape
        # print(out.shape)
        out = torch.index_select(out, 1, idx)
        padding, *_ = self.pad
        k = min(self.kernels) 
        pad = padding - k // 2
        if pad>0:
            out = torch.narrow(out, 2, pad, h - 2 * pad)
            out = torch.narrow(out, 3, pad, w - 2 * pad)
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


def forward_backward_test(big_kernel, small_kernel,b,c,h,w,group_in):
    extra_pad = (small_kernel -1 ) - small_kernel // 2
    nk = math.ceil(big_kernel / small_kernel)
    hout, wout = h-2*extra_pad,w-2*extra_pad
    c_in = c*nk
    # print('cin:',c_in)
    ipt=np.random.rand(b,c_in, h, w).astype(np.float32)
    # ipt=np.ones((b,c_in, h, w)).astype(np.float32)
    # ipt=(np.ones((b,c_in, h, w)) + np.arange(c_in).reshape(1,-1,1,1)).astype(np.float32)
    # ipt=(np.ones((b,c_in, h, w)) + np.arange(w).reshape(1,1,1,-1) + np.arange(c_in).reshape(1,-1,1,1)*20).astype(np.float32)
    # ipt=(np.ones((b,c_in, h, w)) + np.arange(w).reshape(1,1,1,-1)).astype(np.float32)
    # ipt[:,:,:,:w//2]-=1
    # ipt = np.arange(b*c_in* h* w).reshape(b,c_in, h, w).astype(np.float32)
    # print(ipt[0,0])
    target=np.random.rand(b,c,hout, wout).astype(np.float32)
    x0 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    x1 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    x2 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    z = Variable(torch.tensor(target).cuda())
    
    #model 1
    lora = LoRAConvsByRandom_cu_clone(c, big_kernel, small_kernel, group_in)
    lora1_x, lora2_x, small_x = lora(x0, hout, wout)
    print(lora1_x.shape, lora2_x.shape, small_x.shape)
    print('---------------------')
    # model 2
    torch.manual_seed(123)
    asft = AddShift_mp_blur_module(big_kernel, small_kernel, c, c_in, group_in)
    asft.cuda()
    print('=======================')
    # # lora.small
    # asft.idx_identit = asft.idx_identit*0 - 1
    # for ii,s in enumerate(lora.small):
    #     asft.idx_identit[s.long(),ii]=4
    # idx_identit=asft.idx_identit
    lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)
    # np.set_printoptions(linewidth=150, edgeitems=35)
    # print(small_z.detach().cpu().numpy())
    # print(torch.stack([idx_identit.T.float().reshape(-1), x1[0,:,0,0].cpu()]).detach().cpu().numpy())
    # print(lora1_z.shape, lora2_z.shape, small_z.shape)
    print('=======================')
    print(torch.all(torch.abs(lora1_x-lora1_z)<1e-3))
    print(torch.all(torch.abs(lora2_x-lora2_z)<1e-3))
    print(torch.all(torch.abs(small_x-small_z)<1e-3))
    
    # model 3
    torch.manual_seed(123)
    asmp = AddShift_mp_module(big_kernel, small_kernel, c, c_in, group_in)
    asmp.cuda()
    lora1_y, lora2_y, small_y = asmp(x2, b, hout, wout)
    print(torch.all(torch.abs(lora1_y-lora1_z)<1e-3))
    print(torch.all(torch.abs(lora2_y-lora2_z)<1e-3))
    print(torch.all(torch.abs(small_y-small_z)<1e-3))
    print('=======================')
    # print(lora1_x[0,0])
    # print(lora1_z[0,0])
    
    # print(small_x[0,0],small_x.shape)
    # print(small_x[0,1],small_x.shape)
    # print('=======================')
    # print(small_z[0,0],small_z.shape)
    # print(small_z[0,1],small_z.shape)

    N = 5
    with Timer("single path w/o grad"):
        for _ in range(N):
            lora1_x, lora2_x, small_x = lora(x0, hout, wout)
    
    with Timer("multi blur path w/o grad"):
        for _ in range(N):
            lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)

    with Timer("multi path w/o grad"):
        for _ in range(N):
            lora1_y, lora2_y, small_y = asmp(x2, b, hout, wout)
    # Average time cost of single path is 14.3731 ms
    # Average time cost of multi path is 8.8034 ms

    print('=======================')
    N = 5
    with Timer("single path w/ grad"):
        for _ in range(N):
            lora1_x, lora2_x, small_x = lora(x0, hout, wout)
            ans0 = lora1_x + lora2_x + small_x
            l0 = F.mse_loss(ans0, z)
            l0.backward()
            gx0=x0.grad
            # x0.grad.zero_()

    
    with Timer("multi blur path w/ grad"):
        for _ in range(N):
            lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)
            ans1 = lora1_z + lora2_z + small_z
            l1 = F.mse_loss(ans1, z)
            l1.backward()
            gx1=x1.grad
            # x1.grad.zero_()
    
    with Timer("multi path w/ grad"):
        for _ in range(N):
            lora1_y, lora2_y, small_y = asmp(x2, b, hout, wout)
            ans2 = lora1_y + lora2_y + small_y
            l2 = F.mse_loss(ans2, z)
            l2.backward()
            gx2=x2.grad
            # x1.grad.zero_()
            
    diff=torch.abs(gx2-gx1)
    print('grad for max diff:', torch.max(diff), ',mean of diff:', torch.mean(diff[diff>1e-3]))
    print('grad:', torch.all(torch.abs(gx2-gx1)<1e-3))
    return big_kernel, small_kernel, c, c_in, group_in, x2.cpu(), b, hout, wout, lora1_y.cpu(), lora2_y.cpu(), small_y.cpu(), z.cpu(), gx2.cpu(), asmp.pad_hv.cpu(), asmp.idx_identit.cpu(),asmp.idx_out.cpu(),asmp.extra_pad

if __name__ =="__main__":
    print('**********************\n export CUDA_VISIBLE_DEVICES=7') 
    import random
    out={}
    for ii in range(5):
        sk= 3 #random.randint(2,7)//2*2+1
        nk= random.randint(11,53)
        bk= (nk*sk)//2*2+1
        b,c= random.randint(1,9),random.randint(2,30)
        h,w=20,25
        h,w=random.randint(bk+21,bk+40),random.randint(bk+21,bk+40)
        group_in = random.randint(1,9)
        print(bk, sk,b,c,h,w,group_in)
        ii_out=forward_backward_test(bk, sk,b,c,h,w,group_in)
        print(f'~~~~~~~~~~~~~{ii}~~~~~{b}~~~~~~~~~~~~~')
        out[ii]=ii_out
    torch.save(out, 'linear.pth')