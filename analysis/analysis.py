import torch

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


def sparse():
    seg='/home/lili/cc/mmsegmentation/workdirs/t77ms/latest.pth'
    base='/home/lili/cc/SLaK/checkpoints/gr2_300_unirep_v0_gap131_M2_N4_g023_s03/checkpoint-best.pth'
    seg_model=load_checkpoint(seg)
    base_model=load_checkpoint(base)
    for key in base_model:
        if not('LoRAs' in key and 'weight' in key):continue
        key_seg = key
        if key_seg not in seg_model:
            key_seg = 'backbone.' + key_seg
        weight=base_model[key]
        weight_seg=seg_model[key_seg]
        print(f"density of {key} is {cal_density(weight):.3f} -- {cal_density(weight_seg):.3f}")
        

            
sparse()    