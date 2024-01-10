import taichi as ti
import math 
import torch 
import torch.nn.functional as F

# torch.backends.cudnn.benchmark = True
# # turn off TF32 for higher accuracy
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False

# ti.init(arch=ti.cuda, kernel_profiler=True)

@ti.kernel
def shift_add_forward(x: ti.types.ndarray(ndim=4),
        out: ti.types.ndarray(ndim=4),
        idxes: ti.types.ndarray(ndim=2),
        pads: ti.types.ndarray(ndim=1),
        B: ti.i32, C1: ti.i32,   C2: ti.i32,  extra_pad: ti.i32, 
        H: ti.i32, W: ti.i32, H2: ti.i32, W2: ti.i32, isH: ti.i8):
    nk=C2//C1
    for b, c, h, w in ti.ndrange(B, C1, H, W):
        s=0.0
        for i in ti.ndrange(nk):
            pad=pads[i]
            idx=idxes[c,i]
            if isH>0:
                ww=w-pad
                if 0 <= ww<= W2-1: 
                    hh=h+extra_pad
                    ss=x[b,idx,hh,ww]
                    s+=ss
            else:
                hh=h-pad
                if 0 <= hh<= H2-1: 
                    ww=w+extra_pad
                    ss=x[b,idx,hh,ww]
                    s+=ss

        out[b,c,h,w] = s

 

@ti.kernel
def shift_add_backward(gx: ti.types.ndarray(ndim=4),
        gout: ti.types.ndarray(ndim=4),
        idxes: ti.types.ndarray(ndim=2),
        pads: ti.types.ndarray(ndim=1),
        B: ti.i32, C1: ti.i32,   C2: ti.i32,  extra_pad: ti.i32, 
        H: ti.i32, W: ti.i32, H2: ti.i32, W2: ti.i32, isH: ti.i8):
    nk=C2//C1
    for b, c, h, w in ti.ndrange(B, C1, H, W):
        gs=gout[b,c,h,w]
        for i in ti.ndrange(nk):
            pad=pads[i]
            idx=idxes[c,i]
            if isH>0:
                ww=w-pad
                if 0 <= ww<= W2-1: 
                    hh=h+extra_pad
                    gx[b,idx,hh,ww]=gs  
            else:
                hh=h-pad
                if 0 <= hh<= H2-1: 
                    ww=w+extra_pad
                    gx[b,idx,hh,ww]=gs 



class TimeX_Taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idxes, pads, B, C1, C2, H, W, H2, W2, extra_pad,isH):
        ctx.B = B
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.H = H
        ctx.W = W
        ctx.H2 = H2
        ctx.W2 = W2
        ctx.extra_pad = extra_pad
        ctx.isH = isH
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        # w = w.contiguous()
        # k = k.contiguous()
        ctx.save_for_backward(idxes, pads)
        out = torch.empty((B, C1, H, W), device='cuda', 
                         memory_format=torch.contiguous_format)
        shift_add_forward(input, out, idxes, pads, B, C1, C2,  extra_pad, H, W, H2, W2,isH)
        ti.sync()
        return out
 

    @staticmethod
    def backward(ctx, gwk):
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        idxes, pads = ctx.saved_tensors
        gx = torch.empty((ctx.B, ctx.C2, ctx.H2, ctx.W2), device='cuda',
                         memory_format=torch.contiguous_format)*0
        shift_add_backward(gx, gwk.contiguous(), idxes, pads,
                            ctx.B, ctx.C1, ctx.C2,  ctx.extra_pad, 
                            ctx.H, ctx.W, ctx.H2, ctx.W2, ctx.isH)
        ti.sync()
        # return gx
        return (gx, None, None, None, None, None, None, None, None, None, None, None)


def RUN_TAICHI(input, idxes, pads, B, C1, C2, H, W, H2, W2, extra_pad,isH):
    return TimeX_Taichi.apply(input, idxes, pads, B, C1, C2, H, W, H2, W2, extra_pad,isH)
