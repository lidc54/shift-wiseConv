from __future__ import print_function
import torch
import math
import copy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from funcs import redistribution_funcs, growth_funcs, prune_funcs

def snip_core(grad, weight,name=None,net=None):
    return torch.abs(weight*grad)

def snip_core_sum(grad, weight,name=None,net=None):
    _core = torch.abs(weight*grad)
    if len(weight.size()) == 2:
        return _core
    return torch.mean(_core, (2,3))

def snip_core_sum_rep(grad, weight,name=None,named_parameters=None):
    _core = torch.abs(weight*grad) 
    # 开启如下模块会变得各stage的稀疏度非常畸形
    # N_rep=9
    # for i in range(1, N_rep):
    #     rep_name=name.replace('0.weight',f'{i}.weight')
    #     if rep_name not in named_parameters:continue
    #     print('++++++++++',rep_name)
    #     weight_rep = named_parameters[rep_name]
    #     grad_rep = weight_rep.grad
    #     _core_rep = torch.abs(weight_rep*grad_rep)
    #     _core+=_core_rep
    if len(weight.size()) == 2:
        return _core
    return torch.mean(_core, (2,3))

def snip_core_cat_rep(grad, weight,name=None,named_parameters=None):
    _core = torch.abs(weight*grad) 
    if len(weight.size()) == 2:
        return _core
    N_rep=9
    out_core=[torch.mean(_core, (2,3))]
    for i in range(1, N_rep):
        rep_name=name.replace('0.weight',f'{i}.weight')
        if rep_name not in named_parameters:continue
        print('++++++++++',rep_name)
        weight_rep = named_parameters[rep_name]
        grad_rep = weight_rep.grad
        _core_rep = torch.abs(weight_rep*grad_rep)
        out_core.append(torch.mean(_core_rep, (2,3)))
    return torch.cat(out_core, dim=1)

def fix_SNIP_sparse(layer_wise_sparsities, masks):
    new_layer_wise_sparsities = []
    for sparsity_, name in zip(layer_wise_sparsities, masks):
        if 'stages.0' in name: # 0.1 valueable
            sparsity_ = max(1-0.1, sparsity_)
        if 'stages.1' in name:
            sparsity_ = max(1-0.68, sparsity_)
        if 'stages.2' in name:
            sparsity_ = max(1-0.72, sparsity_)
        if 'stages.3' in name: # 0.9 valueable
            sparsity_ = 1-0.9
        new_layer_wise_sparsities.append(sparsity_)
    return new_layer_wise_sparsities

def SNIP(net, keep_ratio, train_dataloader, device, masks, args, gw_fun=snip_core):
    if args.distributed:
        train_dataloader.sampler.set_epoch(0)

    # Grab a single batch from the training dataset
    images, labels = next(iter(train_dataloader))
    input_var = images.to(device, non_blocking=True)
    target_var = labels.to(device, non_blocking=True)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    net.zero_grad()
    outputs = net(input_var)
    loss = F.cross_entropy(outputs, target_var)
    loss.backward()

    grads_abs = []
    named_parameters=dict(net.named_parameters())
    for name, weight in net.named_parameters():
        if name not in masks: continue
        # grads_abs.append(torch.abs(weight*weight.grad))
        grads_abs.append(gw_fun(weight, weight.grad,name,named_parameters))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    layer_wise_sparsities = []
    for g in grads_abs:
        mask = (g > acceptable_score).float()
        sparsity = float((mask==0).sum().item() / mask.numel())
        layer_wise_sparsities.append(sparsity)

    net.zero_grad()
    return layer_wise_sparsities

def default_SNIP(net, keep_ratio, train_dataloader, device, masks, args, gw_fun=snip_core):
    if args.distributed:
        train_dataloader.sampler.set_epoch(0)

    # Grab a single batch from the training dataset
    images, labels = next(iter(train_dataloader))
    input_var = images.to(device, non_blocking=True)
    target_var = labels.to(device, non_blocking=True)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    net.zero_grad()
    outputs = net(input_var)
    loss = F.cross_entropy(outputs, target_var)
    loss.backward()

    pwconv_grads_abs = []
    lk_grads_abs = []
    grads_abs = []
    grads_idx = []
    named_parameters=dict(net.named_parameters())
    for name, weight in net.named_parameters():
        if name not in masks: continue
        # grads_abs.append(torch.abs(weight*weight.grad))
        grad_mark = gw_fun(weight, weight.grad,name,named_parameters)
        grads_abs.append(grad_mark)
        if 'large_kernel.LoRA' in name:
            grads_idx.append(0)
            lk_grads_abs.append(grad_mark)
        else:
            grads_idx.append(1)
            pwconv_grads_abs.append(grad_mark)

    def get_acceptable_score(grads_abs_, keep_ratio):
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs_])
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        return acceptable_score
    
    lk_cceptable_score = get_acceptable_score(lk_grads_abs, keep_ratio)
    if len(pwconv_grads_abs):
        pwconv_cceptable_score = get_acceptable_score(pwconv_grads_abs, keep_ratio)
    else:
        pwconv_cceptable_score = 0
    acceptable_scores = [lk_cceptable_score, pwconv_cceptable_score]

    layer_wise_sparsities = []
    for ii,g in enumerate(grads_abs):
        acceptable_score = acceptable_scores[grads_idx[ii]]
        mask = (g > acceptable_score).float()
        sparsity = float((mask==0).sum().item() / mask.numel())
        layer_wise_sparsities.append(sparsity)

    net.zero_grad()
    return layer_wise_sparsities

class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1, init_step=0):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)
        if init_step!=0:
            for i in range(init_step):
                self.cosine_stepper.step()
    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']


class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)
    """
    def __init__(self, optimizer, train_loader, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='random', use_embed_mask=False, redistribution_mode='momentum', verbose=False, fp16=False, args=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient', 'magnitude_sum', 'random_sum']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))
        print('growth_modes is:', str(growth_mode))
        print('prune_mode is:', str(prune_mode))
        self.args = args
        self.device = torch.device(args.device)
        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose
        self.train_loader = train_loader
        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer
        self.baseline_nonzero = None

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}
        self.prune_rate = prune_rate
        self.steps = 0
        self.half = fp16
        self.name_to_32bit = {}

        # multi path in "rep" mode. all path share same mask
        self.use_rep=False
        self.use_embed_mask=use_embed_mask
        # multi path in "chunk" mode. all path share same mask
        self.use_chunk=False
        self.n_chunk=1

        if self.args.fix:
            self.args.update_frequency = None

    # 后来添加，目的是将使用rep方式的group conv的一层作为一个整体 
    #想法就是mask不加rep， xx=pi_net.state_dict() xx.keys()， 上面的snip中使用statedict
    #最后mask再复制到rep中
    def add_module_sum_rep(self, module):
        self.use_rep=True
        self.modules.append(module)
        self.module = module
        print('￥￥SUM_REP￥￥'*10)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                # if self.args.only_L:
                if 'large_kernel.LoRA' in name:
                    mark=name.split('large_kernel.LoRA')[1]
                    if ('LoRA.weight' not in name) and ('0.weight' not in mark): 
                        continue
                    self.names.append(name)
                    print('---', name)
                    cin, cout, k1, k2 = tensor.shape
                    self.masks[name] = torch.zeros((cin, cout, 1, 1), dtype=torch.float32, requires_grad=False).to(self.device)
                elif not self.args.only_L and 'pwconv' in name:
                    print('+++', name)
                    self.names.append(name)
                    self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        # print('￥￥￥￥￥￥￥￥￥'*10)
        # for name in self.masks: print(name, self.masks[name].shape)
        if self.args.sparse_init == 'snip':
            # self.init_sum_snip(density=1-self.args.sparsity, gw_fun=snip_core_sum_rep)
            self.init_sum_snip(density=1-self.args.sparsity, gw_fun=snip_core_cat_rep)
        
        mask_keys=list(self.masks.keys())

        N_rep=9
        named_parameters=dict(self.module.named_parameters())
        for name in mask_keys:
            for i in range(N_rep):
                rep_name=name.replace('0.weight',f'{i}.weight')
                if rep_name not in named_parameters:continue
                # rep_name=name.replace('split_convs','split_rep_convs')
                data=copy.deepcopy(self.masks[name])
                self.masks[rep_name]=data
        print('￥￥SUM_REP￥￥'*10)
        for name in self.masks: 
            print(name, self.masks[name].shape)

    # 将group conv的每一层作为一个整体，同时chunk为n层。
    def add_module_sum_chunk(self, module,n_chunk):
        self.use_chunk=True
        self.n_chunk=n_chunk
        self.modules.append(module)
        self.module = module
        print('##CHUNK##'*10)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                # if self.args.only_L:
                if 'large_kernel.LoRA' in name:# only conv and bn_bias and bn_weight
                    # if "XX" in name: continue
                    self.names.append(name)
                    print('---', name)
                    cin, cout, k1, k2 = tensor.shape
                    self.masks[name] = torch.zeros((cin//n_chunk, cout, 1, 1), dtype=torch.float32, requires_grad=False).to(self.device)
                elif not self.args.only_L and 'downsample_layers' not in name:
                    print('+++', name)
                    self.names.append(name)
                    self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        print('##CHUNK##'*10,'----', n_chunk)
        for name in self.masks: print(name, self.masks[name].shape)
        if self.args.sparse_init == 'snip':
            self.init_sum_snip(density=1-self.args.sparsity, gw_fun=snip_core_sum)

    # 后来添加，目的是将group conv的每一层作为一个整体
    def add_module_sum(self, module):
        self.modules.append(module)
        self.module = module
        print('@@SUM@@'*10)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                # if self.args.only_L:
                if 'large_kernel.LoRA' in name:# only conv and bn_bias and bn_weight
                    # if "XX" in name: continue
                    self.names.append(name)
                    print('---', name)
                    cin, cout, k1, k2 = tensor.shape
                    self.masks[name] = torch.zeros((cin, cout, 1, 1), dtype=torch.float32, requires_grad=False).to(self.device)
                elif not self.args.only_L and 'downsample_layers' not in name:
                    print('+++', name)
                    self.names.append(name)
                    self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        print('@@SUM@@'*10)
        for name in self.masks: print(name, self.masks[name].shape)
        if self.args.sparse_init == 'snip':
            self.init_sum_snip(density=1-self.args.sparsity, gw_fun=snip_core_sum)

    def init_sum_snip(self, density=0.05, erk_power_scale=1.0, gw_fun=None):
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        self.density = density

        print('initialize by snip')
        self.baseline_nonzero = 0
        # layer_wise_sparsities = SNIP(self.module, density, self.train_loader, self.device, self.masks, self.args, gw_fun=gw_fun)
        layer_wise_sparsities = default_SNIP(self.module, density, self.train_loader, self.device, self.masks, self.args, gw_fun=gw_fun)
        # layer_wise_sparsities = fix_SNIP_sparse(layer_wise_sparsities, self.masks)

        for sparsity_, name in zip(layer_wise_sparsities, self.masks):
            self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1 - sparsity_)).float().data.to(
                self.device)

        masks=copy.deepcopy(self.masks)
        self.masks={}
        keys=list(masks.keys())
        keys.sort(key=lambda x:len(x))
        for k in keys:self.masks[k]=masks[k]

        total_size = 0
        sparse_size = 0
        dense_layers = []
        for name, weight in self.masks.items():
            dense_weight_num = weight.numel()
            sparse_weight_num = (weight != 0).sum().int().item()
            total_size += dense_weight_num
            sparse_size += sparse_weight_num
            layer_density = sparse_weight_num / dense_weight_num
            if layer_density >= 0.99: dense_layers.append(name)
            print(f'Density of layer {name} with tensor {weight.size()} is {layer_density}')
        print('Final sparsity level of {0}: {1}'.format(1-self.density, 1 - sparse_size / total_size))

        # masks of layers with density=1 are removed
        for name in dense_layers:
            self.masks.pop(name)
            print(f"pop out layer {name}")

        self.apply_mask()

    def add_module(self, module):
        self.modules.append(module)
        self.module = module
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                if self.args.only_L:
                    if 'large_kernel.LoRA' in name:
                        self.names.append(name)
                        self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)
                else:
                    self.names.append(name)
                    self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        self.init(mode=self.args.sparse_init, density=1-self.args.sparsity)


    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def init(self, mode='snip', density=0.05, erk_power_scale=1.0):
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        self.density = density

        if mode == 'uniform':
            print('initialized with uniform')
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.to(self.device)
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'resume':
            print('initialized with resume')
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.to(self.device)
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'snip':
            print('initialize by snip')
            self.baseline_nonzero = 0
            layer_wise_sparsities = SNIP(self.module, density, self.train_loader, self.device, self.masks, self.args)

            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1 - sparsity_)).float().data.to(
                    self.device)

        elif mode == 'ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            self.baseline_nonzero = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
                self.baseline_nonzero += weight.numel() * density
            is_epsilon_valid = False

            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros

                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale

                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        total_size = 0
        sparse_size = 0
        dense_layers = []
        for name, weight in self.masks.items():
            dense_weight_num = weight.numel()
            sparse_weight_num = (weight != 0).sum().int().item()
            total_size += dense_weight_num
            sparse_size += sparse_weight_num
            layer_density = sparse_weight_num / dense_weight_num
            if layer_density >= 0.99: dense_layers.append(name)
            print(f'Density of layer {name} with tensor {weight.size()} is {layer_density}')
        print('Final sparsity level of {0}: {1}'.format(1-self.density, 1 - sparse_size / total_size))

        # masks of layers with density=1 are removed
        for name in dense_layers:
            self.masks.pop(name)
            print(f"pop out layer {name}")

        self.apply_mask()

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')


    def step(self):
        self.optimizer.step()
        self.apply_mask()

        # decay the adaptation rate for better results
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        self.steps += 1

        if self.args.update_frequency is not None:
            if self.steps % self.args.update_frequency == 0:
                print('*********************************Dynamic Sparsity********************************')
                self.truncate_weights()
                self.print_nonzero_counts()


    def apply_mask(self):

        # synchronism masks
        if self.args.distributed:
            self.synchronism_masks()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        if self.use_chunk:
                            mask=self.masks[name]
                            mask=mask.unsqueeze(1).repeat(1,self.n_chunk,1,1,1).reshape(-1,1,1,1)
                            tensor.data = tensor.data*mask
                            if 'momentum_buffer' in self.optimizer.state[tensor]:
                                self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*mask
                        else:
                            tensor.data = tensor.data*self.masks[name]
                            if 'momentum_buffer' in self.optimizer.state[tensor]:
                                self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def prun_sub_topk(self, prune_rate, name, weight=None):
        """
        The first branch, in terms of sparsity, has three key features: difference, randomness, and subset, when compared to the other branches
        """
        num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        idx = None
        if weight is not None:
            assert len(weight.data.shape)==4, 'weight should come from a conv'
            w=torch.sum(torch.abs(weight.data),(2,3),keepdim=True)
            if num_remove == 0.0: idx = None

            x, idx = torch.sort(w.view(-1))
        return k, idx
    
    def truncate_weights(self):

        for module in self.modules:
            state_dict=module.state_dict()
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if self.use_rep and (self.steps // self.args.update_frequency % self.args.sparse_gap == 0):
                    if '.0.weight' not in name:continue
                    weight=torch.abs(weight.detach())
                    for ii in range(1,9):
                        rep_name = name.replace('.0.weight', f'.{ii}.weight')
                        if rep_name not in state_dict: continue
                        weight += torch.abs(state_dict[rep_name].detach())
                if self.use_chunk:
                    weight=torch.abs(weight.detach())
                    oc,ic,ks,ks=weight.shape
                    weight=weight.reshape(-1,self.n_chunk,ic,ks,ks).sum(1)
                #         print(f'{ii} ', end='')
                # print('prune-->: ',name)
                if len(weight.data.shape)==2:
                    weight = weight.detach().unsqueeze(2).unsqueeze(3)

                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                # prune
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.name2nonzeros[name] - new_mask.sum().item()
                self.name2removed[name] = removed
                self.masks[name][:] = new_mask
                # copy prune out to other path
                if self.use_rep and (self.steps // self.args.update_frequency % self.args.sparse_gap == 0):
                    if '.0.weight' not in name:continue
                    rep_idx=None
                    for ii in range(1,9):
                        rep_name = name.replace('.0.weight', f'.{ii}.weight')
                        if rep_name not in self.masks: continue
                        if self.use_embed_mask:
                            # prune
                            # sparsity become 0.5 from 0.7 for rep
                            rep_weight = None if ii > 1 else weight
                            rep_mask = self.masks[rep_name]
                            self.name2nonzeros[rep_name] = rep_mask.sum().item()
                            self.name2zeros[rep_name] = rep_mask.numel() - self.name2nonzeros[name]
                            rep_k, rep_idx = self.prun_sub_topk(self.prune_rate*5/3, rep_name, rep_weight) # more prune
                            rep_mask.data.view(-1)[rep_idx[:rep_k]] = 0.0
                            rep_removed = self.name2nonzeros[rep_name] - rep_mask.sum().item()
                            self.name2removed[rep_name] = rep_removed
                            self.masks[rep_name][:] = rep_mask
                        else:
                            self.name2nonzeros[rep_name] = self.name2nonzeros[name]
                            self.name2zeros[rep_name] = self.name2zeros[name]
                            self.name2removed[rep_name] = self.name2removed[name]

                #         print(f'{ii} ', end='')
                # print('prune copy other mask: ',name)

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if self.use_rep and (self.steps // self.args.update_frequency % self.args.sparse_gap == 0) and ('.0.weight' not in name):
                    continue
                new_mask_ = self.masks[name].data.byte()
                # growth
                new_mask = self.growth_func(self, name, new_mask_, math.floor(self.name2removed[name]), weight)
                diff_mask = new_mask_ ^ new_mask # grown part for path zero
                self.masks[name][:] = new_mask.float()
                # copy mask to other path
                if self.use_rep and (self.steps // self.args.update_frequency % self.args.sparse_gap == 0):
                    if '.0.weight' not in name:continue
                    for ii in range(1,9):
                        rep_name = name.replace('.0.weight', f'.{ii}.weight')
                        if rep_name not in self.masks: continue
                        if self.use_embed_mask:
                            rep_mask = self.masks[rep_name].data.byte()
                            diff = (torch.rand(new_mask.shape).cuda() < (5/7)) & diff_mask # part of new grown
                            new_mask_ = rep_mask | diff
                            self.masks[rep_name][:] = new_mask_.float()
                        else:
                            self.masks[rep_name][:] = new_mask.float()
                #         print(f'{ii} ', end='')
                # print('growth-->: ',name)

        self.apply_mask()

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               num_nonzeros / float(mask.numel()))
                print(val)

        print('Prune rate: {0}\n'.format(self.prune_rate))
        if self.use_rep:
            print(f'use rep--> share mask among different path using {self.prune_func.__name__} and {self.growth_func.__name__} | sparse_gap: {self.steps // self.args.update_frequency % self.args.sparse_gap} equal 0 will using same mask for multi-path')

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                # print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    def synchronism_masks(self):

        for name in self.masks.keys():
            torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

