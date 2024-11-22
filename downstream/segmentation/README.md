We use MMSegmentation frameworks. 

1. Just clone MMSegmentation, and
```
git clone https://github.com/open-mmlab/mmsegmentation.git
git checkout v0.30.0

mkdir data
ln -s  ~/cc/dataset/ade20k_2016/ data/ade

pip install prettytable
pip install -U openmim
mim install mmcv-full==1.7.2
vim /root/miniconda3/lib/python3.8/site-packages/mmcv/utils/config.py
line 502 remove verify
vim /root/miniconda3/lib/python3.8/site-packages/numpy/core/function_base.py
line 120 int(num)
```
2. Put ```segmentation/SW_v2_unirep.py``` into ```mmsegmentation/mmseg/models/backbones/```. The only difference between ```segmentation/SW_v2_unirep.py``` and ```SW_v2_unirep.py``` for ImageNet classification is the ```@BACKBONES.register_module```.
3. Add SLaK into ```mmsegmentation/mmseg/models/backbones/__init__.py```. That is

  ```
  ...
 from .SW_v2_unirep import ShiftWise_v2
  __all__ = ['ResNet', ..., 'ShiftWise_v2']
  
  ...
 from .unireplknet import UniRepLKNetBackbone
  __all__ = ['ResNet', ..., 'UniRepLKNetBackbone']
  ```
4. Put ```segmentation/configs/*.py``` into ```mmsegmentation/configs/SW/```; put files of ```mmsegmentation/mmseg/core/optimizers/``` into ```mmsegmentation/mmseg/core/optimizers/```.
<!-- 4. Download and use our weights. For examples, to evaluate SW-tiny + UperNet on ADE20K
  ```
  python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/SW/upernet_sw_tiny_512_80k_ade20k_ss.py --launcher pytorch --eval mIoU
  ``` -->
5. Or you may finetune our released pretrained weights
  ```
  export CUDA_VISIBLE_DEVICES=0,1,2,3
   bash tools/dist_train.sh  configs/SW/upernet_sw_tiny_512_160k_ade20k_ss.py 4 --work-dir ADE20_SW_51_sparse_1000ite/ --auto-resume  --seed 0 --deterministic
   ```
   The path of pretrained models is 'checkpoint_file' in 'upernet_sw_tiny_512_80k_ade20k_ss'.

6. test
```
python tools/test.py workdirs/t77ms/upernet_sw_tiny_512_160k_ade20k_may_ms.py  workdirs/t77ms/latest.pth --eval mIoU

CUDA_VISIBLE_DEVICES=4,5 bash tools/dist_test.sh workdirs/s06_t77best/ss_test.py workdirs/s06_t77best/latest.pth 2  --eval mIoU
```
7. flops
```
python tools/get_flops.py workdirs/t77ms/upernet_sw_tiny_512_160k_ade20k_may_ms.py --shape 512 2048
tiny
    rep1 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 948.12 GFLOPs
    Params: 61.73 M
    ==============================

    rep2 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 953.81 GFLOPs
    Params: 62.51 M
    ==============================
small:rep1 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 1039.21 GFLOPs
    Params: 87.5 M
    ==============================
```