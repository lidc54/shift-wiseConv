# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 4 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=2:  320k iters, 2 batchsize per GPU
#   we use 4 GPUs with 80k schedule:           80k iters, 8 batchsize per GPU

_base_ = [
    './upernet_SW.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

checkpoint_file = '/path/to/checkpoint-best.pth'
dims = [80, 160, 320, 640]
model = dict(
    backbone=dict(
        type='ShiftWise_v2',
        in_chans=3,
        depths=[3, 3, 18, 3], 
        dims=dims,
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        kernel_size=[51,49,47,13,3],
        width_factor=1.0,
        ghost_ratio=0.23,
        N_path=2,
        N_rep=4,
        sparse=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(num_classes=150, in_channels=dims),
    auxiliary_head=dict(num_classes=150, in_channels=dims[2]), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                #  paramwise_cfg={'decay_rate': 0.9,
                #                 'decay_type': 'stage_wise',
                #                 'num_layers': 6})
                paramwise_cfg={'decay_rate': 1,
                                'decay_type': 'layer_wise',
                                'num_layers': 9})

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU; in our case we use 4 gpus and 4 images per GPU)
data=dict(samples_per_gpu=4)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
