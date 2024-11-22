# https://blog.csdn.net/Scabbards_/article/details/129600219


# from backbones import *
import torch
import sys, time
import json
import sys
# Add WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension into your PYTHONPATH by the following commands:
# sys.path.append('WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension')

def get_throughoupt(model, optimal_batch_size=128):
    device = torch.device('cuda')
    for m in model.modules():
        if hasattr(m, 'reparameterize'):
            m.reparameterize()
    model.eval()
    model.to(device)
    model.eval()
    dummy_input = torch.randn(optimal_batch_size, 3,224,224, dtype=torch.float).to(device)
    repetitions=100
    total_time = 0
    
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    Throughput = (repetitions*optimal_batch_size)/total_time
    return Throughput

def get_throughoupt2(model, optimal_batch_size=128):
    device = torch.device('cuda')
    for m in model.modules():
        if hasattr(m, 'reparameterize'):
            m.reparameterize()
    model.eval()
    model.to(device)
    model.eval()
    dummy_input = torch.randn(optimal_batch_size, 3,224,224, dtype=torch.float).to(device)
    repetitions=25
    total_time = 0
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    with torch.no_grad():
        starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for rep in range(repetitions):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        ender.record()
        total_time = starter.elapsed_time(ender)/1000
        # total_time += curr_time
    Throughput = (repetitions*optimal_batch_size)/total_time
    torch.cuda.synchronize()
    time.sleep(0.5)
    torch.cuda.synchronize()
    return Throughput

#############################################################################################################################
#                                             SLaK                                                                          #
#############################################################################################################################
def slak_gemm(throughoupt_fun):
    from models.SLaK_gemm import SLaK_tiny, SLaK_small
    info={}
    for model_name, model in zip(['SLaK_tiny', 'SLaK_small'], [SLaK_tiny, SLaK_small]):
        info[model_name]={}
        for kernel_merged in [False, True]:
            slak_tiny_model = model(pretrained=False, Decom=True, kernel_size=[51, 49, 47, 13, 5], width_factor=1.3, drop_path_rate=0.2, small_kernel_merged=kernel_merged)
            sz_info={}
            for sz in [32,64, 128]:
                try:
                    thr_1 = throughoupt_fun(slak_tiny_model, optimal_batch_size=sz)
                    sz_info[sz]=thr_1
                except Exception as e:
                    print(e)
            info[model_name][f'kernel_merged_{kernel_merged}']=sz_info
    return info

def slak_general(throughoupt_fun):
    from models.SLaK_general import SLaK_tiny, SLaK_small
    info={}
    for model_name, model in zip(['SLaK_tiny', 'SLaK_small'], [SLaK_tiny, SLaK_small]):
        info[model_name]={}
        for kernel_merged in [False, True]:
            slak_tiny_model = model(pretrained=False, Decom=True, kernel_size=[51, 49, 47, 13, 5], width_factor=1.3, drop_path_rate=0.2, small_kernel_merged=kernel_merged)
            sz_info={}
            for sz in [32,64, 128]:
                try:
                    thr_1 = throughoupt_fun(slak_tiny_model, optimal_batch_size=sz)
                    sz_info[sz]=thr_1
                except Exception as e:
                    pass
            info[model_name][f'kernel_merged_{kernel_merged}']=sz_info
    return info
    

#############################################################################################################################
#                                             ShiftWise                                                                     #
#############################################################################################################################

def sw_throughoupt(throughoupt_fun):
    from models.SW_v2_unirep import ShiftWise_v2_tiny, ShiftWise_v2_small
    info={}
    for model_name, model in zip(['ShiftWise_v2_tiny', 'ShiftWise_v2_small'], [ShiftWise_v2_tiny, ShiftWise_v2_small]):
        info[model_name]={}
        for N_rep in range(1,5):
            sw_model = model(pretrained=False, kernel_size=[51, 49, 47, 13, 5], width_factor=1.0, drop_path_rate=0.2, ghost_ratio=0.23, N_path=1, N_rep=N_rep, deploy=False)
            sz_info={}
            for sz in [32, 64, 128]:
                try:
                    thr_1 = throughoupt_fun(sw_model, optimal_batch_size=sz)
                    sz_info[sz]=thr_1
                except Exception as e:
                    pass
            info[model_name][f'N_rep_{N_rep}']=sz_info
    return info

def sw_throughoupt_deploy(throughoupt_fun):
    from models.SW_v2_unirep import ShiftWise_v2_tiny, ShiftWise_v2_small
    info={}
    for model_name, model in zip(['ShiftWise_v2_tiny', 'ShiftWise_v2_small'], [ShiftWise_v2_tiny, ShiftWise_v2_small]):
        info[model_name]={}
        for N_rep in range(1,5):
            sw_model = model(pretrained=False, kernel_size=[51, 49, 47, 13, 5], width_factor=1.0, drop_path_rate=0.2, ghost_ratio=0.23, N_path=1, N_rep=N_rep, deploy=True)
            sz_info={}
            for sz in [32, 64, 128]:
                try:
                    thr_1 = throughoupt_fun(sw_model, optimal_batch_size=sz)
                    sz_info[sz]=thr_1
                except Exception as e:
                    pass
            info[model_name][f'N_rep_{N_rep}']=sz_info
    return info

#############################################################################################################################
#                                             unireplknet                                                                   #
#############################################################################################################################

def uni_throughoupt(throughoupt_fun):
    from models.unireplknet import unireplknet_t, unireplknet_s
    info={}
    for model_name, model in zip(['unireplknet_t', 'unireplknet_s'], [unireplknet_t, unireplknet_s]):
        info[model_name]={}
        for deploy_ in [True, False]:
            uni_model = model(deploy=deploy_)#kernel_size=13, attempt_use_lk_impl=False
            sz_info={}
            for sz in [32,64, 128]:
                try:
                    thr_1 = throughoupt_fun(uni_model, optimal_batch_size=sz)
                    sz_info[sz]=thr_1
                except Exception as e:
                    pass
            info[model_name][f'deploy_{deploy_}']=sz_info
    return info

total={}
for fun_name, fun_ in zip(['onebyone', 'allonce'],[get_throughoupt, get_throughoupt2]):
    if fun_name=='':
        continue
    print(fun_name)
    out={}
    out['slak_gemm']=slak_gemm(fun_)
    out['slak_general']=slak_general(fun_)
    out['SW']=sw_throughoupt(fun_)
    out['SW_deploy']=sw_throughoupt_deploy(fun_)
    out['uni']=uni_throughoupt(fun_)
    total[fun_name] = out

print(json.dumps(total, indent=4))
with open('fps.json','w')as f:
    json.dump(total, f, indent=4)