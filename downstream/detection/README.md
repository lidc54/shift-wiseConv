We use MMDetection frameworks. 

1. Clone MMDetection, and
```
git clone https://github.com/open-mmlab/mmdetection.git
git checkout v2.28.2

pip install scipy
pip install terminaltables
pip install -U cython
pip install git+https://gitee.com/pursuit_zhangyu/cocoapi.git#subdirectory=PythonAPI
pip install numpy==1.23.0
mkdir data
ln -s  ~/cc/dataset/coco/ data/coco
```
2. Put ```mmdetection/mmdet/models/backbones/```. The only difference between ```mmdetection/SW_v2_unirep.py``` and ```SW_v2_unirep.py``` for ImageNet classification is the ```@BACKBONES.register_module```.
3. Add SLaK into ```mmdetection/mmdet/models/backbones/__init__.py``` . That is

  ```
  ...
 from .SW_v2_unirep import ShiftWise_v2
  __all__ = ['ResNet', ..., 'ShiftWise_v2']
  ```
4. Put ```detection/configs/*.py``` into ```mmdetection/configs/SW/```; put files of ```mmdetection/mmdet/core/optimizers/``` into ```mmdetection/mmdet/core/optimizers/```.

5. Or you may finetune our released pretrained weights
  ```
  export CUDA_VISIBLE_DEVICES=2,4,5,6
   bash tools/dist_train.sh  configs/SW/cascade_mask_rcnn_sw_tiny_120_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py  4 --work-dir ctest --auto-resume  --seed 0 --deterministic
  bash tools/dist_train.sh  configs/SW/cascade_mask_rcnn_sw_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py  4 --work-dir ctest77 --auto-resume  --seed 0 --deterministic
   ```
   The path of pretrained models is 'checkpoint_file' in 'upernet_sw_tiny_512_80k_ade20k_ss'.

### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md). [公共数据集](https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/dataset_prepare.html)

6. test
```
python tools/test.py configs/SW/cascade_mask_rcnn_sw_tiny_mstrain_480-800_adamw_3x_coco_in1k.py ctest77/epoch_2.pth --eval bbox segm
```
7. vis COCO


8. [flops](https://mmdetection.readthedocs.io/en/v2.28.0/useful_tools.html#model-complexity)
```
python tools/analysis_tools/get_flops.py work_dirs/tiny_flops.py  --shape 1280 800
tiny
==============================
Input shape: (3, 1280, 800)
Flops: 751.21 GFLOPs
Params: 86.89 M
==============================
small
==============================
Input shape: (3, 1280, 800)
Flops: 839.27 GFLOPs
Params: 110.48 M
==============================
```