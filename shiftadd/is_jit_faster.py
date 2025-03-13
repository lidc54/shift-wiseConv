import torch
import torch.nn as nn
from time import perf_counter
import numpy as np

def timer(f,*args):   
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

 
# 定义一个简单的模块，包含两个并行的分支
class ParallelBranches(nn.Module):
    def __init__(self):
        super(ParallelBranches, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.fxs = nn.ModuleList([
        #     nn.Conv2d(128, 128, 3, 1, 1)
        #     for _ in range(5)
        # ])
 
    def forward(self, x):
        # 并行执行两个分支
        a = self.conv1(x)
        b = self.conv2(x)
        c = self.conv3(x)
        d = self.conv4(x)
        e = self.conv5(x)
        # y=0
        # for fx in self.fxs:
        #     y+=fx(x)
        # 合并结果
        # return y
        return a+b+c+d+e
    
class P2s(torch.jit.ScriptModule):
    def __init__(self):
        super(P2s, self).__init__()
        self.fxs = nn.ModuleList([
            nn.Conv2d(128, 128, 3, 1, 1)
            for _ in range(5)
        ])

    @torch.jit.script_method
    def forward(self, x):
        # 并行执行两个分支
        y=0
        for fx in self.fxs:
            y+=fx(x)
        # 合并结果
        return y 
 
# 使用torch.jit.script来加速模块
model = torch.jit.script(ParallelBranches())
model2 = ParallelBranches()
model3 = P2s()
 
# 示例输入
example = torch.rand(1, 3, 100, 100)
example_gpu = torch.rand(1, 128, 56, 56).cuda()
 
# 前向传播
# output = model(example)
# print(output)
# print(f'pytorch cpu: {np.mean([timer(model2,example) for _ in range(10)])}')
# print(f'torchscript cpu: {np.mean([timer(model,example) for _ in range(10)])}')

model.cuda()
model2.cuda()
model3.cuda()
model3(example_gpu)
model2(example_gpu)
model(example_gpu)
print(f'pytorch gpu: {np.mean([timer(model2,example_gpu) for _ in range(10)])}')
print(f'torchscript gpu: {np.mean([timer(model,example_gpu) for _ in range(10)])}')
print(f'ModuleList gpu: {np.mean([timer(model3,example_gpu) for _ in range(10)])}')


