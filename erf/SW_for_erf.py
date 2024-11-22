# A script to visualize the ERF.
# More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity (https://arxiv.org/pdf/2207.03620.pdf)
# Github source: https://github.com/VITA-Group/SLaK
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/DingXiaoH/RepLKNet-pytorch.
# --------------------------------------------------------'

from backbones.SW_v2_unirep import ShiftWise_v2
checkpoint_file='/path/to/checkpoint-299.pth'
class SWForERF(ShiftWise_v2):

    def __init__(self):
        super().__init__(in_chans=3,
                        depths=[3, 3, 18, 3], 
                        dims=[80, 160, 320, 640], 
                        drop_path_rate=0.4,
                        layer_scale_init_value=1.0,
                        out_indices=[0, 1, 2, 3],
                        kernel_size=[51, 49, 47, 13, 3],
                        width_factor=1.0,
                        ghost_ratio=0.23,
                        N_path=2,
                        N_rep=4,
                        sparse=True,
                        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
                    )
        # The default model here is SLaK-T. Changing dims for SLaK-S/B.
    def forward(self, x):
        # x = self.forward_features(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
        # return self.norm(x)     #   Using the feature maps after the final norm also makes sense. Observed very little difference.

