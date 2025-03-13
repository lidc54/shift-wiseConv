"""
目的是参照AddShift_mp_module, 查看AddShift_mp_linear_module的正确性
"""
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ops import AddShift_mp_module, AddShift_mp_embedding_module
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

def SinPositionEncoding(max_sequence_length, d_model, base=10000):
    pe = torch.zeros(max_sequence_length, d_model, dtype=torch.float)  # size(max_sequence_length, d_model)
    exp_1 = torch.arange(d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
    exp_value = exp_1 / (d_model / 2)

    alpha = 1 / (base ** exp_value)  # size(dmodel/2)
    out = torch.arange(max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  # size(max_sequence_length, d_model/2)
    embedding_sin = torch.sin(out)
    embedding_cos = torch.cos(out)

    pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
    pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
    return pe.transpose(0,1).reshape(-1).contiguous()

def forward_backward_test(big_kernel, small_kernel,b,c,h,w,group_in):
    extra_pad = (small_kernel -1 ) - small_kernel // 2
    nk = math.ceil(big_kernel / small_kernel)
    hout, wout = h-2*extra_pad,w-2*extra_pad
    c_in = c*nk
    # print('cin:',c_in)
    ipt=np.random.rand(b,c_in, h, w).astype(np.float32)
    # ipt=(np.ones((b,c_in, h, w)) + np.arange(c_in).reshape(1,-1,1,1)).astype(np.float32)
    # ipt=(np.ones((b,c_in, h, w)) + np.arange(w).reshape(1,1,1,-1)).astype(np.float32)
    # ipt[:,:,:,:w//2]-=1
    # ipt = np.arange(b*c_in* h* w).reshape(b,c_in, h, w).astype(np.float32)
    # print(ipt[0,0])
    target=np.random.rand(b,c,hout, wout).astype(np.float32)
    x0 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    x1 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    z = Variable(torch.tensor(target).cuda())
    
    # model 2
    torch.manual_seed(123)
    asft = AddShift_mp_module(big_kernel, small_kernel, c, c_in, group_in)
    asft.cuda()
    lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)
    print(lora1_z.shape, lora2_z.shape, small_z.shape)
    print('=======================')

    #model 1
    # 1.1 简单测试
    aa=torch.FloatTensor([[0]*nk for i in range(c)]*group_in).reshape(-1).cuda()
    cc=torch.FloatTensor([0 for i in range(c)]*group_in).reshape(-1).cuda()
    bb=torch.FloatTensor([1 for i in range(c)]).reshape(1,-1,1,1).cuda()
    bc=torch.FloatTensor([[1]*nk for i in range(c)]).reshape(1,-1,1,1).cuda()

    # aa=torch.FloatTensor([[i+1]*nk for i in range(c)]*group_in).reshape(-1).cuda()
    # cc=torch.FloatTensor([i+1 for i in range(c)]*group_in).reshape(-1).cuda()
    # bb=torch.FloatTensor([i+1 for i in range(c)]).reshape(1,-1,1,1).cuda()
    # bc=torch.FloatTensor([[i+1]*nk for i in range(c)]).reshape(1,-1,1,1).cuda()
    # 1.2 位置编码
    aa = SinPositionEncoding(group_in, c_in, base=10000)
    cc = SinPositionEncoding(group_in, c, base=10000)

    torch.manual_seed(123)
    lora = AddShift_mp_embedding_module(big_kernel, small_kernel, c, c_in, group_in,w1=aa,w2=aa,w3=cc)
    lora.cuda()
    #*******************************
    lora1_x, lora2_x, small_x = lora(x0, b, hout, wout)
    print(lora1_x.shape, lora2_x.shape, small_x.shape)
    print('---------------------')

    print(torch.all(torch.abs(lora1_x/bb-lora1_z)<1e-3))
    print(torch.all(torch.abs(lora2_x/bb-lora2_z)<1e-3))
    print(torch.all(torch.abs(small_x/bb-small_z)<1e-3))# false
    #这里可以验证前两个分支是一致的

    # torch.all(torch.abs(small_x[:,0]-small_z[:,0]))
    # if not(torch.all(torch.abs(lora1_x-lora1_z)<1e-3) && \
    #        torch.all(torch.abs(lora2_x-lora2_z)<1e-3) &&\
    #        torch.all(torch.abs(small_x-small_z)<1e-3)):
    #     SystemExit()
    print('=======================')
    # print(lora1_x[0,0])
    # print(lora1_z[0,0])
    
    # print(small_x[0,0],small_x.shape)
    # print(small_x[0,1],small_x.shape)
    # print('=======================')
    # print(small_z[0,0],small_z.shape)
    # print(small_z[0,1],small_z.shape)

    N = 1
    with Timer("multi path linear"):
        for _ in range(N):
            lora1_x, lora2_x, small_x = lora(x0, b, hout, wout)
    
    with Timer("multi path"):
        for _ in range(N):
            lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)
    # Average time cost of multi path linear is 0.1602 ms
    # Average time cost of multi path is 0.0927 ms


    print('forward=======================')
    N = 1
    with Timer("multi path linear"):
        for _ in range(N):
            lora1_x, lora2_x, small_x = lora(x0, b, hout, wout)
            ans0 = lora1_x + lora2_x + small_x
            l0 = F.mse_loss(ans0, z)
            l0.backward()
            gx0=x0.grad
            # x0.grad.zero_()

    
    with Timer("multi path"):
        for _ in range(N):
            lora1_z, lora2_z, small_z = asft(x1, b, hout, wout)
            ans1 = lora1_z + lora2_z + small_z
            l1 = F.mse_loss(ans1, z)
            l1.backward()
            gx1=x1.grad
            # x1.grad.zero_()
    
    print('grad=======================')        
    # gx0[0,0,0,0] #lora.idx_out # aa[:,0,:,10,10]
    # 这儿是修改了cuda里的代码，只保留small分支，结果就是和权重bb/bc是完全一致的
    tt=torch.index_select(gx0, 1, lora.idx_identit.reshape(-1))
    tt=torch.max(torch.max(tt, 3)[0],2)[0][0] == bb.reshape(-1)
     
    # b,c1,h,w=gx0.shape
    # ss=torch.max(torch.max(torch.max(gx0.reshape(b,-1,nk,h,w),4)[0],3)[0], 2)[1][0]
    # tt=torch.arange(c).cuda()*nk+ss
    diff=torch.abs(gx0/bc-gx1)
    print('grad for max:', torch.max(diff), ',mean:', torch.mean(diff[diff>1e-3]))
    print('grad:', torch.all(torch.abs(gx0/bc-gx1)<1e-3))

if __name__ =="__main__":
    print('**********************\n export CUDA_VISIBLE_DEVICES=7') 
    import random
    for ii in range(1):
        bk= random.randint(7,53)//2*2+1
        sk= random.randint(2,7)//2*2+1
        b,c=random.randint(1,9),random.randint(2,30)
        h,w=random.randint(bk+21,bk+40),random.randint(bk+21,bk+40)
        group_in = 1 #random.randint(1,4)
        print(bk, sk,b,c,h,w,group_in)
        forward_backward_test(bk, sk,b,c,h,w,group_in)
        print(f'~~~~~~~~~~~~~{ii}~~~~~{b}~~~~~~~~~~~~~')