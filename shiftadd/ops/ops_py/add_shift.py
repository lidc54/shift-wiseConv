'''
# https://blog.csdn.net/qq_42722197/article/details/135944397
# 1. single path --16
# 2. multi path multi rep --50
# 3. multi path multi rep with linear function --187
# 4. multi path multi rep with embedding --351
'''

# ops/ops_py/sum.py
import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import addshift, addshiftmp, addshiftmp_linear, addshiftmp_em, addshiftmp_blur

__all__ = ['AddShift_ops', 'AddShift_mp_module', 'AddShift_mp_linear_module', 'AddShift_mp_embedding_module', 'AddShift_mp_blur_module']

# 1. single path
class AddShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, idxes, pads, extra_pad,B, C_in, H_in, W_in, C_out, H_out, W_out, isH):
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H_in = H_in
        ctx.W_in = W_in
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.extra_pad = extra_pad
        ctx.isH = isH
        inputs = inputs.contiguous()
        ctx.save_for_backward(idxes, pads)
        out = torch.empty((B, C_out, H_out, W_out), device='cuda', 
                         memory_format=torch.contiguous_format) 
        addshift.forward(out,inputs,idxes,pads,extra_pad, B, C_in,  H_in, W_in, C_out, H_out, W_out,isH)
        return out


    @staticmethod
    def backward(ctx, gout):
        idxes, pads = ctx.saved_tensors
        ginput = torch.empty((ctx.B, ctx.C_in, ctx.H_in, ctx.W_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gout = gout.contiguous()
        addshift.backward(gout, ginput, idxes, pads,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.isH)
        return (ginput, None, None,None, None, None, None, None, None, None, None, None, None)

AddShift_ops=AddShift.apply

# 2. multi path multi rep
class AddShiftMp(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in, H_in, W_in, C_out, H_out, W_out, group_in):
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H_in = H_in
        ctx.W_in = W_in
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.extra_pad = extra_pad
        ctx.group_in = group_in
        group_out = 3
        ctx.group_out = group_out
        inputs = inputs.contiguous()
        ctx.save_for_backward(pad_hv, idx_identit, idx_out)
        # print('pad_hv', pad_hv,'\n',
        #       'idx_identit', idx_identit, '\n',
        #       'idx_out', idx_out, '\n',
        #       'extra_pad',extra_pad,'\n',
        #       'B', B, '\n',
        #       'C_in',  C_in, '\n',
        #       'H_in', H_in, '\n',
        #       'W_in',  W_in,'\n',
        #       'C_out',  C_out, '\n',
        #       'H_out',  H_out,'\n',
        #       'W_out', W_out, '\n',
        #       'group_in',group_in)
        out = torch.empty((B*group_out, C_out, H_out, W_out), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        addshiftmp.forward(out,inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in,  H_in, W_in, C_out, H_out, W_out, group_in)
        return out
 

    @staticmethod
    def backward(ctx, gout):
        pad_hv, idx_identit, idx_out = ctx.saved_tensors
        ginput = torch.empty((ctx.B, ctx.C_in, ctx.H_in, ctx.W_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gout = gout.contiguous()
        addshiftmp.backward(gout, ginput, pad_hv, idx_identit, idx_out,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.group_in)
        return (ginput, None, None,None, None, None, None, None, None, None, None, None, None)

AddShift_mp_ops=AddShiftMp.apply


class AddShift_mp_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in) -> None: 
        '''
        c_out:repN
        c_in:out_n
        group_in:N_rep 有四组index
        '''
        super().__init__()
        self.device = None
        self.c_in = c_in
        self.c_out = c_out
        self.group_in = group_in
        
        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        self.extra_pad, pad_hv, idx_identit = self.shuffle_idx_2_gen_pads(small_kernel, c_out, group_in)
        idx_out = torch.IntTensor([[i] * self.nk for i in range(c_out)]).reshape(-1)
        # print(self.extra_pad)
        # print(self.pad_hv)
        # print(self.idx_identit)
        # print('idxout:', self.idx_out)
        # self.register_buffer('pad_hv1', pad_hv)
        # self.register_buffer('idx_identit1', idx_identit)
        # self.register_buffer('idx_out1', idx_out)
        self.pad_hv=pad_hv
        self.idx_identit=idx_identit
        self.idx_out=idx_out

        #todolist:需要注意把输出三通道放到w附近，还是放到channel附近，要尝试哪一个更合适！！！

    def shuffle_idx_2_gen_pads(self, small_kernel, c_out, group_in):
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        extra_pad = padding - small_kernel // 2
        torch.manual_seed(123)
        shuffle_idx_horizon = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_vertica = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_identit = [(torch.randint(0,self.nk,[c_out])+torch.arange(c_out)*self.nk).int() for _ in range(group_in)]
        pad_horizon, pad_vertica = [], []

        # print('shuffle_idx_horizon',shuffle_idx_horizon)
        # print('shuffle_idx_vertica',shuffle_idx_vertica)
        # print('shuffle_idx_identit',shuffle_idx_identit)

        # merge idxes & pads as new pads sort by input
        idx_offset=torch.IntTensor(real_pad*c_out)
        for i in range(group_in):
            idx_horizon = shuffle_idx_horizon[i]
            idx_vertica = shuffle_idx_vertica[i]
            _, indices_h = torch.sort(idx_horizon)
            _, indices_v = torch.sort(idx_vertica)
            pad_horizon.append(torch.index_select(idx_offset, 0, indices_h))
            pad_vertica.append(torch.index_select(idx_offset, 0, indices_v))
        pad_horizon, pad_vertica = torch.stack(pad_horizon), torch.stack(pad_vertica)
        pad_hv = torch.vstack([pad_horizon, pad_vertica])
        pad_hv = torch.transpose(pad_hv, 0, 1)
        shuffle_idx_identit = torch.transpose(torch.stack(shuffle_idx_identit), 0, 1)
        return extra_pad, pad_hv.contiguous(), shuffle_idx_identit.contiguous()
     
    def forward(self, x, b, hout, wout):
        x_hin, x_win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        if self.device is None:
            device = x.device
            self.device=device
            self.pad_hv = self.pad_hv.to(self.device)
            self.idx_identit = self.idx_identit.to(self.device)
            self.idx_out = self.idx_out.to(self.device)

        out = AddShift_mp_ops(x, self.pad_hv, self.idx_identit, self.idx_out,
                              self.extra_pad, b, self.c_in, x_hin, x_win,
                              self.c_out, hout, wout, self.group_in)
        x,y,z = torch.chunk(out, 3, dim=0) 
        return x,y,z
    
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


# 3. multi path multi rep with linear function
class AddShiftMp_linear(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in, H_in, W_in, C_out, H_out, W_out, group_in,w1,w2,w3):
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H_in = H_in
        ctx.W_in = W_in
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.extra_pad = extra_pad
        ctx.group_in = group_in
        group_out = 3
        ctx.group_out = group_out
        inputs = inputs.contiguous()
        ctx.save_for_backward(inputs, pad_hv, idx_identit, idx_out, w1,w2,w3)
        # print('pad_hv', pad_hv,'\n',
        #       'idx_identit', idx_identit, '\n',
        #       'idx_out', idx_out, '\n',
        #       'extra_pad',extra_pad,'\n',
        #       'B', B, '\n',
        #       'C_in',  C_in, '\n',
        #       'H_in', H_in, '\n',
        #       'W_in',  W_in,'\n',
        #       'C_out',  C_out, '\n',
        #       'H_out',  H_out,'\n',
        #       'W_out', W_out, '\n',
        #       'group_in',group_in)
        out = torch.empty((B*group_out, C_out, H_out, W_out), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        addshiftmp_linear.forward(out,inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in,  H_in, W_in, C_out, H_out, W_out, group_in,w1,w2,w3)
        return out
 

    @staticmethod
    def backward(ctx, gout):
        inputs, pad_hv, idx_identit, idx_out, w1,w2,w3 = ctx.saved_tensors
        ginput = torch.empty((ctx.B, ctx.C_in, ctx.H_in, ctx.W_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gout = gout.contiguous()
        addshiftmp_linear.backward(gout, ginput, pad_hv, idx_identit, idx_out,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.group_in,w1,w2,w3)
        # w1.requires_grad==True
        gw1 = torch.empty((ctx.C_in * ctx.group_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gw2 = torch.empty((ctx.C_in * ctx.group_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gw3 = torch.empty((ctx.C_out * ctx.group_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        addshiftmp_linear.w_grad(gout, inputs, pad_hv, idx_identit, idx_out,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.group_in, gw1, gw2, gw3)
        
        return (ginput, None, None,None, None, None, None, None, None, None, None, None, None, gw1, gw2, gw3)

AddShiftMp_linear_ops=AddShiftMp_linear.apply



class AddShift_mp_linear_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in, w1=None, w2=None, w3=None) -> None: 
        '''
        c_out:repN
        c_in:out_n
        group_in:N_rep 比如有四组index
        '''
        super().__init__()
        # self.device = None
        self.c_in = c_in
        self.c_out = c_out
        self.group_in = group_in
        if not w1:
            # path 1
            self.w1 = nn.Parameter(torch.empty((c_in * group_in), memory_format=torch.contiguous_format)) 
            # path 2
            self.w2 = nn.Parameter(torch.empty((c_in * group_in), memory_format=torch.contiguous_format)) 
            # small
            self.w3 = nn.Parameter(torch.empty((c_out * group_in), memory_format=torch.contiguous_format)) 
        else:
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3

        
        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        self.extra_pad, pad_hv, idx_identit = self.shuffle_idx_2_gen_pads(small_kernel, c_out, group_in)
        idx_out = torch.IntTensor([[i] * self.nk for i in range(c_out)]).reshape(-1)
        # print(self.extra_pad)
        # print(self.pad_hv)
        # print(self.idx_identit)
        # print('idxout:', self.idx_out)
        self.register_buffer('pad_hv', pad_hv)
        self.register_buffer('idx_identit', idx_identit)
        self.register_buffer('idx_out', idx_out)
        self.reset_parameters()

        #todolist:需要注意把输出三通道放到w附近，还是放到channel附近，要尝试哪一个更合适！！！

    def reset_parameters(self):
        # https://blog.csdn.net/panbaoran913/article/details/125069375
        for p in self.parameters():
            # if p.dim() > 1:
            #     nn.init.xavier_uniform_(p)    # Xavier 初始化确保权重“恰到好处” 
            # else:
            #     nn.init.uniform_(p)
            nn.init.constant(p, 1.0)

    def shuffle_idx_2_gen_pads(self, small_kernel, c_out, group_in):
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        extra_pad = padding - small_kernel // 2
        # torch.manual_seed(123)
        shuffle_idx_horizon = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_vertica = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_identit = [(torch.randint(0,self.nk,[c_out])+torch.arange(c_out)*self.nk).int() for _ in range(group_in)]
        pad_horizon, pad_vertica = [], []

        # print('shuffle_idx_horizon',shuffle_idx_horizon)
        # print('shuffle_idx_vertica',shuffle_idx_vertica)
        # print('shuffle_idx_identit',shuffle_idx_identit)

        # merge idxes & pads as new pads sort by input
        idx_offset=torch.IntTensor(real_pad*c_out)
        for i in range(group_in):
            idx_horizon = shuffle_idx_horizon[i]
            idx_vertica = shuffle_idx_vertica[i]
            _, indices_h = torch.sort(idx_horizon)
            _, indices_v = torch.sort(idx_vertica)
            pad_horizon.append(torch.index_select(idx_offset, 0, indices_h))
            pad_vertica.append(torch.index_select(idx_offset, 0, indices_v))
        pad_horizon, pad_vertica = torch.stack(pad_horizon), torch.stack(pad_vertica)
        pad_hv = torch.vstack([pad_horizon, pad_vertica])
        pad_hv = torch.transpose(pad_hv, 0, 1)
        shuffle_idx_identit = torch.transpose(torch.stack(shuffle_idx_identit), 0, 1)
        return extra_pad, pad_hv.contiguous(), shuffle_idx_identit.contiguous()
     
    def forward(self, x, b, hout, wout):
        hin, win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        # if self.device is None:
        #     self.hin, self.win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        #     device = x.device
        #     self.device=device
        #     self.pad_hv = self.pad_hv.to(self.device)
        #     self.idx_identit = self.idx_identit.to(self.device)
        #     self.idx_out = self.idx_out.to(self.device)

        out = AddShiftMp_linear_ops(x, self.pad_hv, self.idx_identit, self.idx_out,
                                    self.extra_pad, b, self.c_in, hin, win,
                                    self.c_out, hout, wout, self.group_in,
                                    self.w1, self.w2, self.w3)
        x,y,z = torch.chunk(out, 3, dim=0) 
        return x,y,z
    
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


# 4. multi path multi rep with embedding
class AddShiftMp_embedding(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in, H_in, W_in, C_out, H_out, W_out, group_in,w1,w2,w3):
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H_in = H_in
        ctx.W_in = W_in
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.extra_pad = extra_pad
        ctx.group_in = group_in
        group_out = 3
        ctx.group_out = group_out
        inputs = inputs.contiguous()
        ctx.save_for_backward(pad_hv, idx_identit, idx_out)
        # print('pad_hv', pad_hv,'\n',
        #       'idx_identit', idx_identit, '\n',
        #       'idx_out', idx_out, '\n',
        #       'extra_pad',extra_pad,'\n',
        #       'B', B, '\n',
        #       'C_in',  C_in, '\n',
        #       'H_in', H_in, '\n',
        #       'W_in',  W_in,'\n',
        #       'C_out',  C_out, '\n',
        #       'H_out',  H_out,'\n',
        #       'W_out', W_out, '\n',
        #       'group_in',group_in)
        out = torch.empty((B*group_out, C_out, H_out, W_out), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        addshiftmp_em.forward(out,inputs, pad_hv, idx_identit, idx_out, extra_pad, B, C_in,  H_in, W_in, C_out, H_out, W_out, group_in,w1,w2,w3)
        return out
 

    @staticmethod
    def backward(ctx, gout):
        pad_hv, idx_identit, idx_out = ctx.saved_tensors
        ginput = torch.empty((ctx.B, ctx.C_in, ctx.H_in, ctx.W_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gout = gout.contiguous()
        addshiftmp_em.backward(gout, ginput, pad_hv, idx_identit, idx_out,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.group_in)
        return (ginput, None, None,None, None, None, None, None, None, None, None, None, None, None, None, None)

AddShiftMp_embedding_ops=AddShiftMp_embedding.apply


class AddShift_mp_embedding_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in, w1=None, w2=None, w3=None) -> None: 
        '''
        c_out:repN
        c_in:out_n
        group_in:N_rep 比如有四组index
        '''
        super().__init__()
        # self.device = None
        self.c_in = c_in
        self.c_out = c_out
        self.group_in = group_in
        self.register_buffer('w1', w1) # path 1
        self.register_buffer('w2', w2) # path 2
        self.register_buffer('w3', w3) # small

        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        self.extra_pad, pad_hv, idx_identit = self.shuffle_idx_2_gen_pads(small_kernel, c_out, group_in)
        idx_out = torch.IntTensor([[i] * self.nk for i in range(c_out)]).reshape(-1)
        # print(self.extra_pad)
        # print(self.pad_hv)
        # print(self.idx_identit)
        # print('idxout:', self.idx_out)
        self.register_buffer('pad_hv', pad_hv)
        self.register_buffer('idx_identit', idx_identit)
        self.register_buffer('idx_out', idx_out)
        self.reset_parameters()

        #todolist:需要注意把输出三通道放到w附近，还是放到channel附近，要尝试哪一个更合适！！！

    def reset_parameters(self):
        # https://blog.csdn.net/panbaoran913/article/details/125069375
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    # Xavier 初始化确保权重“恰到好处” 
            else:
                nn.init.uniform_(p)

    def shuffle_idx_2_gen_pads(self, small_kernel, c_out, group_in):
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        extra_pad = padding - small_kernel // 2
        # torch.manual_seed(123)
        shuffle_idx_horizon = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_vertica = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_identit = [(torch.randint(0,self.nk,[c_out])+torch.arange(c_out)*self.nk).int() for _ in range(group_in)]
        pad_horizon, pad_vertica = [], []

        # print('shuffle_idx_horizon',shuffle_idx_horizon)
        # print('shuffle_idx_vertica',shuffle_idx_vertica)
        # print('shuffle_idx_identit',shuffle_idx_identit)

        # merge idxes & pads as new pads sort by input
        idx_offset=torch.IntTensor(real_pad*c_out)
        for i in range(group_in):
            idx_horizon = shuffle_idx_horizon[i]
            idx_vertica = shuffle_idx_vertica[i]
            _, indices_h = torch.sort(idx_horizon)
            _, indices_v = torch.sort(idx_vertica)
            pad_horizon.append(torch.index_select(idx_offset, 0, indices_h))
            pad_vertica.append(torch.index_select(idx_offset, 0, indices_v))
        pad_horizon, pad_vertica = torch.stack(pad_horizon), torch.stack(pad_vertica)
        pad_hv = torch.vstack([pad_horizon, pad_vertica])
        pad_hv = torch.transpose(pad_hv, 0, 1)
        shuffle_idx_identit = torch.transpose(torch.stack(shuffle_idx_identit), 0, 1)
        return extra_pad, pad_hv.contiguous(), shuffle_idx_identit.contiguous()
     
    def forward(self, x, b, hout, wout):
        hin, win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        # if self.device is None:
        #     self.hin, self.win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        #     device = x.device
        #     self.device=device
        #     self.pad_hv = self.pad_hv.to(self.device)
        #     self.idx_identit = self.idx_identit.to(self.device)
        #     self.idx_out = self.idx_out.to(self.device)
        #     self.w1 = self.w1.to(self.device) # path 1
        #     self.w2 = self.w2.to(self.device) # path 2
        #     self.w3 = self.w3.to(self.device) # small

        out = AddShiftMp_embedding_ops(x, self.pad_hv, self.idx_identit, self.idx_out,
                                    self.extra_pad, b, self.c_in, hin, win,
                                    self.c_out, hout, wout, self.group_in,
                                    self.w1, self.w2, self.w3)
        x,y,z = torch.chunk(out, 3, dim=0) 
        return x,y,z
    
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



# 5. multi rep with third path large kernel
class AddShiftMpBlur(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, inputs, pad_hv, idx_identit, idx_out, small_kernel, extra_pad, B, C_in, H_in, W_in, C_out, H_out, W_out, group_in):
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H_in = H_in
        ctx.W_in = W_in
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.extra_pad = extra_pad
        ctx.group_in = group_in
        ctx.small_kernel = small_kernel
        group_out = 3
        ctx.group_out = group_out
        inputs = inputs.contiguous()
        ctx.save_for_backward(pad_hv, idx_identit, idx_out)
        # print('pad_hv', pad_hv,'\n',
        #       'idx_identit', idx_identit, '\n',
        #       'idx_out', idx_out, '\n',
        #       'extra_pad',extra_pad,'\n',
        #       'B', B, '\n',
        #       'C_in',  C_in, '\n',
        #       'H_in', H_in, '\n',
        #       'W_in',  W_in,'\n',
        #       'C_out',  C_out, '\n',
        #       'H_out',  H_out,'\n',
        #       'W_out', W_out, '\n',
        #       'group_in',group_in)
        out = torch.empty((B*group_out, C_out, H_out, W_out), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        addshiftmp_blur.forward(out,inputs, pad_hv, idx_identit, idx_out, small_kernel, extra_pad, B, C_in,  H_in, W_in, C_out, H_out, W_out, group_in)
        return out
 

    @staticmethod
    def backward(ctx, gout):
        pad_hv, idx_identit, idx_out = ctx.saved_tensors
        ginput = torch.empty((ctx.B, ctx.C_in, ctx.H_in, ctx.W_in), device='cuda',
                         memory_format=torch.contiguous_format).zero_()
        gout = gout.contiguous()
        addshiftmp_blur.backward(gout, ginput, pad_hv, idx_identit, idx_out, ctx.small_kernel,
                            ctx.extra_pad, ctx.B, ctx.C_in, ctx.H_in, ctx.W_in,
                            ctx.C_out, ctx.H_out, ctx.W_out, ctx.group_in)
        return (ginput, None, None,None, None, None, None, None, None, None, None, None, None, None)

AddShift_mp_blur_ops=AddShiftMpBlur.apply


class AddShift_mp_blur_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in) -> None: 
        '''
        c_out:repN
        c_in:out_n
        group_in:N_rep 有四组index
        '''
        super().__init__()
        self.device = None
        self.c_in = c_in
        self.c_out = c_out
        self.group_in = group_in
        
        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        self.small_kernel = small_kernel
        self.extra_pad, pad_hv, idx_identit = self.shuffle_idx_2_gen_pads(small_kernel, c_out, group_in)
        idx_out = torch.IntTensor([[i] * self.nk for i in range(c_out)]).reshape(-1)
        self.register_buffer('pad_hv1', pad_hv)
        self.register_buffer('idx_identit1', idx_identit)
        self.register_buffer('idx_out1', idx_out)
        self.pad_hv=pad_hv
        self.idx_identit=idx_identit
        self.idx_out=idx_out
        # print(self.extra_pad)
        # print(self.pad_hv)
        # print(self.idx_identit)
        # print('idxout:', self.idx_out)

        #todolist:需要注意把输出三通道放到w附近，还是放到channel附近，要尝试哪一个更合适！！！

    def shuffle_idx_2_gen_pads(self, small_kernel, c_out, group_in):
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        extra_pad = padding - small_kernel // 2
        # torch.manual_seed(123)
        shuffle_idx_horizon = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        shuffle_idx_vertica = [torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).int() for _ in range(group_in)]
        # shuffle_idx_identit = [(torch.randint(0,self.nk,[c_out])+torch.arange(c_out)*self.nk).int() for _ in range(group_in)]
        pad_horizon, pad_vertica = [], []

        # print('shuffle_idx_horizon',shuffle_idx_horizon)
        # print('shuffle_idx_vertica',shuffle_idx_vertica)
        # print('shuffle_idx_identit',shuffle_idx_identit)
        identity_idxs=[]#创建一个列表，这个列表里有9个有效数字，其他为-1
        for _ in range(group_in):
            tmp_identity_idxs=[]
            for i in range(c_out):
                identity_idx=(torch.zeros(self.nk)-1).long()
                if self.nk>9:
                    i_idx=np.random.choice(self.nk,9,replace=False).tolist()
                    i_idx=torch.LongTensor(i_idx)
                    identity_idx[i_idx]=torch.arange(9)
                else:
                    i_idx=np.random.choice(self.nk, 4, replace=False).tolist()
                    identity_idx[i_idx]=torch.LongTensor([0, 2, 6, 8]) # mid point of 9
                tmp_identity_idxs.append(identity_idx)
            identity_idxs.append(torch.cat(tmp_identity_idxs).int())
            # identity_idxs.append(torch.IntTensor([-1,-1,8,-1,-1,-1,-1,-1,-1]))
        identity_idxs = torch.transpose(torch.stack(identity_idxs), 0, 1)

        # merge idxes & pads as new pads sort by input
        idx_offset=torch.IntTensor(real_pad*c_out)
        for i in range(group_in):
            idx_horizon = shuffle_idx_horizon[i]
            idx_vertica = shuffle_idx_vertica[i]
            _, indices_h = torch.sort(idx_horizon)
            _, indices_v = torch.sort(idx_vertica)
            pad_horizon.append(torch.index_select(idx_offset, 0, indices_h))
            pad_vertica.append(torch.index_select(idx_offset, 0, indices_v))
        pad_horizon, pad_vertica = torch.stack(pad_horizon), torch.stack(pad_vertica)
        pad_hv = torch.vstack([pad_horizon, pad_vertica])
        pad_hv = torch.transpose(pad_hv, 0, 1)
        # shuffle_idx_identit = torch.transpose(torch.stack(shuffle_idx_identit), 0, 1)
        return extra_pad, pad_hv.contiguous(), identity_idxs.contiguous()
     
    def forward(self, x, b, hout, wout):
        x_hin, x_win = hout + 2*self.extra_pad, wout + 2*self.extra_pad
        if self.device is None:
            device = x.device
            self.device=device
            self.pad_hv = self.pad_hv.to(self.device)
            self.idx_identit = self.idx_identit.to(self.device)
            self.idx_out = self.idx_out.to(self.device)

        out = AddShift_mp_blur_ops(x, self.pad_hv, self.idx_identit, self.idx_out,
                              self.small_kernel, self.extra_pad, b, self.c_in, x_hin, x_win,
                              self.c_out, hout, wout, self.group_in)
        x,y,z = torch.chunk(out, 3, dim=0) 
        return x,y,z
    
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

