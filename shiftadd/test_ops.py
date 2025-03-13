from ops import AddShift_ops
import torch
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


if __name__ == '__main__':
    import math
    big_kernel, small_kernel=15,5
    # sw = LoRAConvsByRandom(big_kernel, small_kernel)
    b,c,h,w=2,5,25,25
    nk = math.ceil(big_kernel / small_kernel)
    # padding, pads = sw.pad
    padding, pads = 4,[2,-3,-8]
    x = torch.rand(b,c*nk,h+padding,w+padding).cuda()
    y = torch.zeros(b,c,h,w).cuda()
    # y=sw(x,h,w)
    print('**********************\n export CUDA_VISIBLE_DEVICES=7')
    # y1=(y*0).cuda()
    # x1=copy.copy(x).cuda()
    h_pad = padding - small_kernel // 2
    b, c_in, hin, win = b, c*nk,h+padding,w+padding
    c_out, hout, wout = c,h,w

    # add_v0
    idxes=torch.arange(c*nk).cuda()
    shift_pads=torch.IntTensor(pads).cuda()

    with Timer("AddShift_ops"):
        ans = AddShift_ops(y,x,idxes,shift_pads,h_pad,b, c_in, hin, win, c_out, hout, wout,1)

