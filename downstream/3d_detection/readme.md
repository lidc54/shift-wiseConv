# install
[installation](https://gitee.com/monkeycc/mmdetection3d/blob/v1.0.0rc4/docs/zh_cn/getting_started.md)
```
conda create --name open-mmlab python=3.8
conda install pytorch==1.13.0 torchvision==0.14.0  pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install openmim
mim install mmcv-full==1.5.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc1 (v1.0.0rc4 need mmcv-full==1.6.0)
pip install -v -e .
pip install numba==0.56.4  numpy==1.23.4
pip install timm
```

# Modify the file
- numpy
  - `import numpy;  numpy.__file__` record the pah, eg. /root/miniconda3/lib/python3.8/site-packages/
  - cd to path and `vim /root/miniconda3/lib/python3.8/site-packages/numpy/core/function_base.py`
  - line 120 int(num)
- mmdet
  - optimizers
    - `import mmdet;  mmdet.__file__` record the pah, eg. '/opt/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/mmdet/__init__.py'
    - cd to path and `vim /opt/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/mmdet/core/optimizers/layer_decay_optimizer_constructor.py`
    - line 116
      `raise NotImplementedError() ` of both 'layer_wise' & 'stage_wise'
      change to
      ```
      layer_id = get_layer_id_for_convnext(
          name, self.paramwise_cfg.get('num_layers'))
      logger.info(f'set param {name} as id {layer_id}')
      ```
  - device
    - mmdet/apis/train.py, Line 154, add: cfg.device='cuda'
- mmcv
  - mmcv/utils/config.py(get path as the way shown as above)
  - line 502:  unexpected keyword argument 'verify'




# dataset (follow this [url](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/data_preparation.html#nuscenes))
eg.
```
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz &
wget -c https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz &

wget -c https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs_camera.tgz &ls
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz &

```
- autodl locat in `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`
- unzip
```
tar -xvf  v1.0-trainval02_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval04_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval06_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval08_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval10_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval01_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval03_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval05_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval07_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval09_blobs.tgz -C /dev/shm/nuscene &
tar -xvf  v1.0-trainval_meta.tgz    -C /dev/shm/nuscene &
```
# run
1. create_data
```
cd mmdetection3d/data
ln -s /dev/shm/nuscene/ nuscenes
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

2. Put ```3d_detection/SW_v2_unirep.py``` into ```mmdetection3d/mmdet3d/models/backbones/```. The only difference between ```3d_detection/SW_v2_unirep.py``` and ```SW_v2_unirep.py``` for ImageNet classification is the ```@BACKBONES.register_module```.

3. Add SLaK into ```mmdetection3d/mmdet3d/models/backbones/__init__.py```. That is
  ```
  ...
 from .SW_v2_unirep import ShiftWise_v2
  __all__ = ['ResNet', ..., 'ShiftWise_v2']


  ...
 from .unireplknet import UniRepLKNetBackbone
  __all__ = ['ResNet', ..., 'UniRepLKNetBackbone']
  ```

4. Put ```3d_detection/configs/*.py``` into ```mmdetection3d/configs/SW/```; 

5. [change multi_gpu_test to single_gpu_test](https://github.com/open-mmlab/mmdetection/issues/9744): 
```
vim /opt/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/mmdet/apis/train.py: 247
#eval_hook = DistEvalHook if distributed else EvalHook
eval_hook = EvalHook
```

6. Or you may [finetune our released pretrained weights](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/1_exist_data_model.html#train-with-multiple-gpus)
```

# [test](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/2_new_data_model.html#test-and-inference)
[Evaluation](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/datasets/nuscenes_det.html#evaluation)
```
python tools/test.py configs/SW/fcos3d_sw-tiny_fpn_head-gn_8xb2-1x_nus-mono3d.py work_dirs/fcos3d_sw-tiny_fpn_head-gn_8xb2-1x_nus-mono3d/epoch_2.pth --eval bbox mAP

# multi-gpu may have error in data prepare
CUDA_VISIBLE_DEVICES=5,6 bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8 --eval bbox mAP 
```

# for debug
TORCH_DISTRIBUTED_DEBUG=DETAIL  ./tools/dist_train.sh configs/SW/fcos3d_sw-tiny_fpn_head-gn_8xb2-1x_nus-mono3d.py 2 
./tools/dist_train.sh configs/SW/fcos3d_sw-tiny_fpn_head-gn_8xb2-1x_nus-mono3d.py 8
```
  
  - **or** [DataContainer' object does not support indexing](https://github.com/deepinsight/insightface/issues/1733#issuecomment-1706047986)
```
File "/opt/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/mmdet/models/detectors/base.py", line 137, in forward_test
img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
#----------
img_meta.data[0][img_id]['batch_input_shape'] = tuple(img.size()[-2:])
instead of: img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
```