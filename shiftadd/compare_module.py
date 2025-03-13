import os, sys
print(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.SLaK_gr2_sw_sf_se_mp import LoRAConvsByRandom_cu as oriLora
from models.SLaK_gr2_sw_sf_se_mp_cp import LoRAConvsByRandom_cu as Lora
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time


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


def test(bb,cc,hh,ww,bk,sk):
    conv1 = oriLora(cc,cc,bk,sk,bn=False).cuda()
    conv2 =  Lora(cc,cc,bk,sk,bn=False).cuda()

    conv2.load_state_dict(conv1.state_dict())
    # state=conv1.state_dict()
    # for key in state:
    #     print(key,state[key].shape)
    #     state[key] = state[key] *0 + 1
    # conv1.load_state_dict(state)
    # conv2.load_state_dict(state)

    ipt=np.random.rand(bb,cc,hh,ww).astype(np.float32)

    target=np.random.rand(bb,cc,hh,ww).astype(np.float32)

    x0 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)
    x1 = Variable(torch.tensor(ipt).cuda(), requires_grad=True)

    z = Variable(torch.tensor(target).cuda())
    gr=0.23
    grcc=cc-int(cc*gr)
    target0=np.random.rand(bb,grcc,hh,ww).astype(np.float32)
    z1 = Variable(torch.tensor(target0).cuda())

    # 开始测试精度

    y0,_ = conv1(x0)
    y1,_ = conv2(x1)
    
    print('out same? ',torch.all(torch.abs(y0-y1)<1e-3))
    # print(y0.shape,y1.shape)
    # 开始测试速度

    print('==========infer speed=============')
    N = 5
    with Timer("single path"):
        for _ in range(N):
            y0 = conv1(x0)

    with Timer("multi path"):
        for _ in range(N):
            y1 = conv2(x1)
            
    # Average time cost of single path is 9.4337 ms
    # Average time cost of multi path is 6.4659 ms
    print('==========for/back-ward speed=============')
    N = 5
    with Timer("single path"):
        for _ in range(N):
            # y0 = conv1(x0)
            # l0 = F.mse_loss(y0, z)
            y0,g0 = conv1(x0)
            l0 = F.mse_loss(y0, z1)

            l0.backward()
            gx0=x0.grad

    with Timer("multi path"):
        for _ in range(N):
            # y1 = conv2(x1)
            # l1 = F.mse_loss(y1, z)
            y1,g1 = conv2(x1)
            l1 = F.mse_loss(y1, z1)
            l1.backward()
            gx1=x1.grad
    diff=torch.abs(gx0-gx1)
    print('grad for max:', torch.max(diff), ',mean:', torch.mean(diff[diff>1e-3]))
    print('grad:', torch.all(torch.abs(gx0-gx1)<1e-3))
    # Average time cost of single path is 44.1189 ms
    # Average time cost of multi path is 14.2434 ms
    # 没有cat ghost之前，梯度是一致的，cat之后不一致
    # grad: tensor(False, device='cuda:0')
    print('==========cat difference=============')
    # x0.grad.zero_()
    # x1.grad.zero_()
    # y0,g0 = conv1(x0)
    # y1,g1 = conv2(x1)
    # y11=torch.cat([y1.contiguous(),g1.contiguous()],dim=1)
    # y00=torch.cat([y0.contiguous(),g0.contiguous()],dim=1)
    
    # l0 = F.l1_loss(y00, z)
    # l1 = F.l1_loss(y11, z)

    # l0.backward()
    # l1.backward()
    # gx11=x1.grad
    # gx00=x0.grad
    # print(conv1.ghost)
    # diff1=torch.abs(gx00-gx11)
    # print(torch.mean(diff1,(0, 2,3)))
    # print('grad for max:', torch.max(diff1), ',mean:', torch.mean(diff1[diff1>1e-3]))
    # print('grad:', torch.all(torch.abs(gx00-gx11)<1e-3))
    
    # print('==========cat target difference=============')
    # x0.grad.zero_()
    # x1.grad.zero_()
    # y0,g0 = conv1(x0)
    # y1,g1 = conv2(x1)
    # y11=torch.cat([y1.contiguous(),g1.contiguous()],dim=1)
    # y00=torch.cat([y0.contiguous(),g0.contiguous()],dim=1)
    # z2_0=torch.cat([z1, torch.index_select(x0, 1, conv1.ghost).detach()],dim=1)
    # z2_1=torch.cat([z1, torch.index_select(x1, 1, conv2.ghost).detach()],dim=1)
    
    # l0 = F.l1_loss(y00, z2_0)
    # l1 = F.l1_loss(y11, z2_1)

    # l0.backward()
    # l1.backward()
    # gx11=x1.grad
    # gx00=x0.grad
    # print(conv1.ghost)
    # diff1=torch.abs(gx00-gx11)
    # print(torch.mean(diff1,(0, 2,3)))
    # print('grad for max:', torch.max(diff1), ',mean:', torch.mean(diff1[diff1>1e-3]))
    # print('grad:', torch.all(torch.abs(gx00-gx11)<1e-3))

    # 啊哈哈哈哈，mse-loss会取sum，被mse耽误的一上午

 
if __name__ =="__main__":
    print('**********************\n export CUDA_VISIBLE_DEVICES=7') 
    print('*******notice: add` return x,ghost_inputs` before `x=torch.cat([x,ghost_inputs],dim=1)`*******') 

    import random
    for ii in range(20):
        bk= random.randint(7,53)//2*2+1
        sk= 3 #random.randint(2,7)//2*2+1
        b,c=random.randint(1,9),random.randint(20,90)
        h,w=random.randint(bk+21,bk+40),random.randint(bk+21,bk+40)
        # group_in = random.randint(1,9)
        print(b,c,h,w,bk, sk)
        test(b,c,h,w,bk, sk)
        # catdiff(b,c,h,w,bk, sk)
        print(f'~~~~~~~~~~~~{ii}~~~~~{b}~~~~~~~~~~~~~~')