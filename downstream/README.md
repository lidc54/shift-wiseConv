# Downstream tasks including semantic segmentation on ADE20K, object detection/segmentation on MS COCO

## Env
The conda env is the same for detection and segmentation, but it is different from mmdet3d

## COCO
- mmdetection v2.28.2
- A series of files must be relocated in accordance with the given [instructions](downstream\detection\README.md).
- coco2017


## ADE20K
- decomposing
- mmsegmentation v0.30.0
- A series of files must be relocated in accordance with the given [instructions](downstream\segmentation\README.md).
- ade20k_2016

If the following issue arises, use this method to disregard the warning.
```
multiprocessing/semaphore_tracker.py:144: UserWarning: semaphore_tracker: There appear to be 4 leaked semaphores to clean up at shutdown len(cache))
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
```

Attention: The `img_ratios` parameter in `MultiScaleFlipAug` acts as the switch to enable or disable multi-scale testing.

