import torch
from collections import defaultdict, Counter
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np


def cal_density(weight):
    return (weight != 0).sum().item() / weight.numel()

def load_checkpoint(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu')
    print(ckpt.keys())
    if 'state_dict' in ckpt:
        _state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        _state_dict = ckpt['model']
    else:
        _state_dict = ckpt
    return _state_dict

def mapIndex2random(weight, nk, c_out, group_in=4):
    torch.manual_seed(123)
    shuffle_idx_horizon = [[torch.randperm(nk) for i in range(c_out)] for _ in range(group_in)]
    shuffle_idx_vertica = [[torch.randperm(nk) for i in range(c_out)] for _ in range(group_in)]
    shuffle_idx_identit = [torch.randint(0,nk,[c_out]) for _ in range(group_in)]
    out = []
    for c in range(len(weight)):
        idx = torch.where(weight[c]==0)[0] #_,idx = torch.where(weight==0)
        for g in range(group_in):
            map_h=shuffle_idx_horizon[g][c]
            map_v=shuffle_idx_vertica[g][c]
            map_i=shuffle_idx_identit[g][c]
            out.extend([map_h[idx],map_v[idx]])
            # if map_i in idx:
            #     out.append(map_i.unsqueeze(0))
    out=torch.cat(out)
    return out


def sparse(tomap=False, n_path=4):
    kernel_sizes={
        'stages.0':51,
        'stages.1':49,
        'stages.2':47,
        'stages.3':13
    }
    all_s, all_P = 0, 0
    star_s, star_p= 0,0
    allsp ={}
    star, star_list ={}, []
    state={}
    stage_len={
        'stages.0':3,
        'stages.1':3,
        'stages.2':18,
        'stages.3':3}
    base='/home/lili/cc/SLaK/checkpoints/gr2_300_unirep_v0_gap131_M2_N4_g023_s03/checkpoint-best.pth'
    base_model=load_checkpoint(base)
    for key in base_model:
        if not('LoRAs' in key and '.0.weight' in key):continue
        weight=base_model[key]
        K=0
        for x in kernel_sizes:
            if x in key:
                K=kernel_sizes[x]
                break
        s = (weight != 0).sum().item()
        p = weight.numel()
        all_s += s
        all_P += p
        print(f"density of {key} is {(s/p):.3f}")
        for ii in range(1,9):
            rep_key = key.replace('.0.weight', f'.{ii}.weight')
            if rep_key not in base_model: continue
            rep_weight = base_model[rep_key]
            s = (rep_weight != 0).sum().item()
            p = rep_weight.numel()
            all_s += s
            all_P += p
            print(f"density of {rep_key} is {(s/p):.3f}")
        w=torch.sum(torch.abs(weight.data),(1,2,3),keepdim=False)
        ss = (w != 0).sum().item()
        sp = w.numel()
        star_s += ss
        star_p += sp
        star[key]=[ss,sp]
        star_list.append(1-ss/sp)
        w=w.reshape(-1,math.ceil(K/3))
        state[key]=w
    print(f'sparse of fine:{(all_s/all_P):.3f}')
    print(f'sparse of coarse:{(star_s/star_p):.3f}')
    plt.rcParams['font.size'] = 18
    # along depth
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 4])
    axs=[fig.add_subplot(gs[0, 0:2]),# sparsity
         fig.add_subplot(gs[0, 2]),#stage1
         fig.add_subplot(gs[0, 3]),#stage2
         fig.add_subplot(gs[1, :3]),#stage3
         fig.add_subplot(gs[1, 3]),#stage4
    ]
    # plt.subplot(1, len(kernel_sizes)+1, 1)
    x = np.arange(len(star_list))+1
    ax=axs[0]
    ax.plot(x, star_list)
    kk_len=0
    # ax=plt.gca()
    for k in stage_len:
        k_len=stage_len[k]+kk_len
        alpha=0.6 if '1' not in k else 0.9
        ax.fill_between(x, star_list, where=(x > kk_len) & (x <= k_len), alpha=alpha, label=k)
        kk_len=k_len
    ax.set_xlabel('layer No.', labelpad=-45)
    ax.set_ylabel('sparsity')
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.set_title('(a)')
    # ax.set_xticks(x)
    # ax.text(1, 0, '1', va='center', ha='right', color='red')
    # ax.text(max(x), 0, f'{max(x)}', va='center', ha='left', color='red')
    # ax.legend(bbox_to_anchor=(5, 0), loc='lower left')
    ax.legend()
    titles_stage=['(b)', '(c)','(d)','(e)']

    # along stage
    stages=defaultdict(list)
    CC={}
    for key in state:
        k = ''
        NG=0
        for x in kernel_sizes:
            if x in key:
                k=x
                NG=math.ceil(kernel_sizes[x]/3)
                break
        if not k: continue
        w=state[key]
        if k not in CC:CC[k]=len(w)
        if tomap:
            idx=mapIndex2random(w, NG, len(w), group_in=n_path)
        else:
            ci,idx = torch.where(w==0)
        stages[k].append(idx)
    for ii,k in enumerate(stages):
        ax=axs[ii+1]
        cmap = mpl.cm.get_cmap('viridis', len(stages[k])).reversed()
        channel_num=CC[k]
        for jj,c in enumerate(stages[k]):
            num = math.ceil(kernel_sizes[k]/3)
            ct=Counter(c.cpu().tolist())
            data=[0]*num
            for d,v in ct.items():
                try:
                    if tomap:
                        data[d]=v/(channel_num*n_path*2)
                    else:
                        data[d]=v/(channel_num)
                except Exception as e:
                    print('############',e)
            # from 0 start: no sparse
            x = np.arange(1, num+1)
            ax.plot(x, data,label=f'L{jj}', ls='-', color=cmap(jj))
        ax.set_xlabel('filter No.', labelpad=-45)
        ax.set_xlim(left=1)
        # ax.set_xticks(x)
        # ax.text(1, 0, '1', va='center', ha='right', color='red')
        # ax.text(max(x), 0, f'{max(x)}', va='center', ha='left', color='red')
        # ax.set_ylim(bottom=0)
        if ii in [2]: ax.set_ylabel('sparsity')
        ax.set_title(f'{titles_stage[ii]}{k}')
        ncols=3 if ii==2 else 1
        ax.legend(ncols=ncols)
    # for ii,k in enumerate(stages):
    #     data=torch.cat(stages[k]).reshape(-1)
    #     num = math.ceil(kernel_sizes[k]/3)
    #     # histc = torch.histc(data.float(), bins=(num), min=0, max=num - 1)
    #     ax=axs[1+ii]
    #     # plt.subplot(1, len(kernel_sizes)+1, 2+ii)
    #     ax.hist(data, bins=num, color='skyblue', edgecolor='black')
    #     ax.set_xlabel('group No.')
    #     if ii in [0,2]: plt.ylabel('sparsity')
    #     ax.set_title(k)
    # plt.tight_layout()
    plt.show()

    # along channel
    channels=defaultdict(list)
    for key in state:
        k = ''
        # NG=0
        for x in kernel_sizes:
            if x in key:
                k=x
                # NG=math.ceil(kernel_sizes[x]/3)
                break
        if not k: continue
        w=state[key]
        c_num = torch.sum(w==0,(1))
        channels[k].append(c_num)
        
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 5])
    axs=[fig.add_subplot(gs[0, 0]),#stage1
         fig.add_subplot(gs[0, 1]),#stage2
         fig.add_subplot(gs[1, :]),#stage3
         fig.add_subplot(gs[0, 2]),#stage4
    ]
    
    titles_stage=['(a)', '(b)', '(c)','(d)','(e)']
    for ii,k in enumerate(channels):
        ax=axs[ii]
        cmap = mpl.cm.get_cmap('viridis', len(channels[k])).reversed()
        channel_num=CC[k]
        for jj,c in enumerate(channels[k]):
            num = math.ceil(kernel_sizes[k]/3)+1
            ct=Counter(c.cpu().tolist())
            data=[0]*num
            for d,v in ct.items():
                try:
                    data[d]=v/channel_num
                except Exception as e:
                    print('@@@@@@@@@@@@')
            ax.plot(data,label=f'L{jj}', ls='-', color=cmap(jj))
        # ax.set_xlabel('groups')
        if ii in [0,2]: ax.set_ylabel('sparsity')
        ax.set_title(k)
        ncols=3 if ii==2 else 1
        ax.set_xlabel('pruned num', labelpad=-45)
        ax.set_title(f'{titles_stage[ii]}{k}')
        ax.legend(ncols=ncols)
    plt.show()
            
sparse(tomap=False)