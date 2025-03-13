from SLaK_gr_sw_sf_se import LoRAConvsByRandom, SLaK_gr_swsf_se_tiny
from SLaK_gr_sw_sf_se2 import LoRAConvsByRandom_cu, SLaK_gr_swsf_se_cu_tiny
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

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


dim=128
in_channels, out_channels = dim, dim
bn=True
big_kernel, small_kernel=13,5
groups=dim
stride=1
use_small_conv=True

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)
    elif type(layer) == nn.SyncBatchNorm:
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif type(layer) == nn.BatchNorm2d:
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        
 

lrc1 = LoRAConvsByRandom(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,
                                 group=groups, stride=stride,
                                 use_small_conv=use_small_conv)

lrc2 = LoRAConvsByRandom_cu(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,
                                 group=groups, stride=stride,
                                 use_small_conv=use_small_conv)
lrc2.apply(init_weights)
lrc1.apply(init_weights)
l1w=lrc1.state_dict()
# l2w=lrc2.state_dict()
lrc2.load_state_dict(l1w)
 
lrc1.cuda()
lrc2.cuda()
x11=torch.rand(2,128,67,67)

target=np.random.rand(2,128,67,67).astype(np.float32)
x21=x11.clone()
x11=Variable(x11.cuda(), requires_grad=True)
x21=Variable(x21.cuda(), requires_grad=True)
x1=0.5 + x11
x2=0.5 + x21

z1 = Variable(torch.tensor(target).cuda())
z2 = Variable(torch.tensor(target).cuda())

y0=lrc1(x1)
y0=lrc2(x1)

with Timer("pytorch"):
    # print(x1.grad,'^^^^^^^^')
    y1=lrc1(x1)
    l0 = F.mse_loss(y1, z1)
    l0.backward()
    gx0=x11.grad

with Timer("cu"):
    y2=lrc2(x2)
    l1 = F.mse_loss(y2, z2)
    l1.backward()
    gx1=x21.grad

# print('x1',x1.shape,x1.sum().data,x1.mean().data)
# print('x2',x2.shape,x2.sum().data,x2.mean().data)
print('----------------------------')
print('y1',y1.shape,y1.sum().data,y1.mean().data)
print('y2',y2.shape,y2.sum().data,y2.mean().data)
print('----------------------------')
yy=torch.abs(y1-y2)
gx=torch.abs(gx1-gx0)
print('diff',yy.sum().data,yy.mean().data,'******', gx.sum().data,gx.mean().data)
time.sleep(1)