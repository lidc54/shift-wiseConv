import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
from matplotlib import pyplot as plt, cm


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



    
# 分析查看有哪些区域有效的把信息传递到了下一层
class LoraCover(nn.Module):
    '''
    cover area of the module
    '''

    def __init__(self,
                #  in_channels: int,
                #  out_channels: int,
                big_kernel, small_kernel,
                stride=1, group=1,
                bn=True, use_small_conv=True):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        self.small_conv = use_small_conv
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        
        self.rep= 1
        

    def forward(self, ori_h, ori_w, use_rand=False, plot_distri=False, repN=None):
        if repN is None:
            repN= 1
        if use_rand:
            torch.manual_seed(123)
            lora1s = [torch.cat([torch.randperm(self.nk) for i in range(repN)]).int() for _ in range(self.rep)]#+ i * self.nk
            lora2s = [torch.cat([torch.randperm(self.nk) for i in range(repN)]).int() for _ in range(self.rep)]# + i * self.nk
        else:
            lora1s = [torch.cat([torch.arange(self.nk-1,-1,-1).int() for i in range(repN)]).int() for _ in range(self.rep)]# + i * self.nk
            lora2s = [torch.cat([torch.arange(self.nk).int() for i in range(repN)]).int() for _ in range(self.rep)]# + i * self.nk
        smalls = [(torch.randint(0,self.nk,[repN])).int() for _ in range(self.rep)]#+torch.arange(repN)*self.nk

        padding, real_pad = self.pad
        x = torch.zeros(1, self.nk, ori_h+padding*2, ori_w+padding*2)
        for ii in range(self.rep):
            for r in range(repN):
                out_idx_H=self.forward_lora(ori_h+padding*2, ori_w+padding*2, ori_h, ori_w, idx=lora1s[ii][self.nk * r: self.nk * (r+1)], VH='H')
                out_idx_W=self.forward_lora(ori_h+padding*2, ori_w+padding*2, ori_h, ori_w, idx=lora2s[ii][self.nk * r: self.nk * (r+1)], VH='W')
                # if self.small_conv:
                #     y += self.forward_small(x, self.small)#, bn_small
                for i in range(self.nk):
                    cx,ys,ye,xs,xe=out_idx_H[i]
                    x[0,cx,ys:ye,xs:xe]+=1
                    cx,ys,ye,xs,xe=out_idx_W[i]
                    x[0,cx,ys:ye,xs:xe]+=1
                si = smalls[ii][r:r+1].data.tolist()[0]
                print(si, padding,(ori_h+padding))
                x[0,si,padding:(ori_h+padding), padding:(ori_w+padding)]+=1

        plt.rc('font',family='Times New Roman') 
        # plt.rcParams.update({'font.size': 18})
        # 画覆盖分布图
        if plot_distri:
            # return x
            data=x.int().numpy()
            m=math.ceil(self.nk/3.0)
            plt.figure(figsize=(15,m*4+1))
            
            cmap = plt.cm.viridis
            fig, axes = plt.subplots(3,m,sharex=True,sharey=True)
            vmin=data.min()
            vmax=data.max()
            for i in range(3*m):
                mm=i//m
                nn=i%m
                if i>=self.nk:
                    fig.delaxes(axes[mm,nn])
                    continue
                ax=axes[mm,nn]
                dd=data[0,i]
                pp=ax.imshow(dd, cmap=cmap, vmin=vmin, vmax=vmax) 
                r=1.0*np.sum(dd==0)/((ori_h+padding)*(ori_w+padding))
                mv=dd.mean()
                if r>1.0:
                    print()
                ax.set_title(f'{int(100*(1-r))}%', fontsize=16, y=0.3)#m{mv:.1f}
            
            # plt.subplot(2,6,12)
            # plt.legend(fontsize=48, loc='upper right')
            os.makedirs('cover',exist_ok=True)
            ax=axes[2, m-1]
            fig.colorbar(pp, ax=ax, shrink=0.5, pad=0.05)
            fig.tight_layout()
            plt.suptitle('areaof(0) & mean v')
            plt.savefig(f'cover/hw{ori_h}x{ori_w}_R{use_rand}.png') 
            return None
        else: # 画曲线图
            data=x.int().numpy()
            return data

    def forward_small(self, out, idx): #, small_bn):
        # shift along the index of every group
        b, c, h, w = out.shape
        out = torch.index_select(out, 1, idx)
        padding, *_ = self.pad
        k = min(self.kernels) 
        pad = padding - k // 2
        if pad>0:
            out = torch.narrow(out, 2, pad, h - 2 * pad)
            out = torch.narrow(out, 3, pad, w - 2 * pad)
        # if self.use_bn:
        #     out = small_bn(out)
        return out

    def forward_lora(self, h, w, ori_h, ori_w, VH='H', idx=None, bn=None):
        out_idx=[]
        for i in range(self.nk):
            out_idx.append(self.rearrange_data(idx[i], h, w, i, ori_h, ori_w, VH))
        return out_idx

    #添加的padding
    def rearrange_data(self, c,h, w, idx, ori_h, ori_w, VH):
        padding, pads = self.pad
        k = min(self.kernels)
        pad=pads[idx] 

        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        # need to calculate start point after conv
        # how many windows shift from real start window index
        s = max(0-pad, 0)
        pads = max(pad,0)
        # padding on other direction
        yy = padding - k // 2
        if VH == 'H':
            # assume add sufficient padding for origin conv
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1-pads
            return c,yy,h - 2 * yy, s,s+suppose_len
        else:
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1-pads
            return c,s,s+suppose_len, yy, w - 2 * yy

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

def plot_trend(data):
    out={}
    for ii,kernel in enumerate(data):
        data2=data[kernel]
        kernel=int(kernel.split('_')[1])
        # if kernel not in [51,13]:
        #     continue
        xys={}
        for userand in data2:
            data3=data2[userand]
            userand=int(userand.split('_')[1])
            xs=[]
            y_mean, y_cover = [],[]
            for x,y in data3.items():
                xs.append(x)
                y_mean.append(y.mean())
                y=y>0
                y_cover.append(y.mean())
            xys[userand]=[xs,y_mean,y_cover]
        out[kernel]=xys
    
    # plot line
    print()
    # plt.figure(figsize=(8, 8))
    plt.rc('font',family='Times New Roman') 
    plt.rcParams.update({'font.size': 20})
    markers=[['*','v'],['d','o'], ['x','s'], ['$\Omega$','$\circledR$']]
    colors=['r','g','b','#f19790']
    legends={0:'w/o',1:'w/'}
    for k in range(2):
        for ii,kernel in enumerate(out):
            mm=markers[ii]
            c=colors[ii]
            # for k in xys:
            xys=out[kernel]
            xs,y_mean,y_cover = xys[k]
            plt.plot(xs,y_cover, label=f'{legends[k]}:M={kernel}', marker=mm[k],color=c, alpha = 0.5)
    
    
    plt.title("Feature utilization rate")     
    plt.xlabel("N paths")
    plt.ylabel("rate")  
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.legend()
    plt.show()
    
    
if __name__ =="__main__":
    print('**********************\n export CUDA_VISIBLE_DEVICES=7') 
    sps=[[[56, 56], (3, 51)],
    [[28, 28], (3, 49)],
    [[14, 14], (3, 47)],
    [[7, 7], (3, 13)]]
    big_kernel, small_kernel=51,5
    h,w=61,61
    useRenN=[1,2,4,8]
    useRenN=[1]
    # plot_distri : will plot cover
    plot_distri=True
    out={}
    for (h,w),(small_kernel, big_kernel) in sps:
        nk = math.ceil(big_kernel / small_kernel)
        sw = LoraCover(big_kernel, small_kernel)
        datas={}
        for use_rands in [0,1]:
            rep_datas={}
            for reps in useRenN:
                data=sw(h, w, use_rands, repN=reps, plot_distri=plot_distri)
                rep_datas[reps]=data
            datas[f'userands_{use_rands}']=rep_datas
        out[f'k_{big_kernel}']=datas
        
        
    if not plot_distri:
        plot_trend(out)