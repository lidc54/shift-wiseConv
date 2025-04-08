# (CVPR2025)ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect
### [Arxiv](https://arxiv.org/abs/2401.12736)  |  [code](https://github.com/lidc54/shift-wiseConv)

**TL;DR:** Our research finds that $3 \times 3$ convolutions can replace larger ones in CNNs, enhancing performance and echoing VGG's results. It also introduces novel parameter settings that have not been previously explored.

<p align="center">
<img src="SW.png" width="500" height="320">
</p>

**Abstract:**
Large kernels make standard convolutional neural networks (CNNs) great again over transformer architectures in various vision tasks. 
Nonetheless, recent studies meticulously designed around increasing kernel size have shown diminishing returns or stagnation in performance. Thus, the hidden factors of large kernel convolution that affect model performance remain unexplored. In this paper, we reveal that the key hidden factors of large kernels can be summarized as two separate components: extracting features at a certain granularity and fusing features by multiple pathways. To this end, we leverage the multi-path long-distance sparse dependency relationship to enhance feature utilization via the proposed Shiftwise (SW) convolution operator with a pure CNN architecture. In a wide range of vision tasks such as classification, segmentation, and detection, SW surpasses state-of-the-art transformers and CNN architectures, including SLaK and UniRepLKNet. More importantly, our experiments demonstrate that $3 \times 3$ convolutions can replace large convolutions in existing large kernel CNNs to achieve comparable effects, which may inspire follow-up works.


## Installation

The code is tested used CUDA 11.7, cudnn 8.2.0, PyTorch 1.10.0.

Create an new conda virtual environment
```
conda create -n shiftWise python=3.8 -y
conda activate shiftWise
```

Install [Pytorch](https://pytorch.org/)>=1.10.0. For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install timm tensorboardX six
```
Install module:
```
cd shiftadd
python setup.py install
```

Train：
```
python -m torch.distributed.run  --master_port=29501 --nproc_per_node=8 main.py  --sparse --width_factor 1.0 -u 100 --sparsity 0.3 --warmup_epochs 5 --sparse_init snip --prune_rate 0.3 --growth random --sparse_type rep_coarse --epochs 300 --model ShiftWise_v2_tiny  --drop_path 0.2 --batch_size 64  --lr 4e-3 --update_freq 8 --model_ema false --model_ema_eval false  --data_path /root/autodl-tmp/imagenet --num_workers 64 --kernel_size 51 49 47 13 3 --only_L  --ghost_ratio 0.23 --sparse_gap 131 --output_dir checkpoints/ 2>&1 |tee tiny.log;
```


## Results and ImageNet-1K trained models

 **name**   | **resolution** | **acc@1**       | **log**       | **model**                                                                                          
:----------:|:--------------:|:---------------:|:-------------:|:-------------:
 **SW-tiny** | 224x224        | 83.39(300epoch)  | [SW-T](backbones/SW_300_unirep_tiny_gap131_M2_N4_g023_s03.log) | [Google Drive](https://drive.google.com/file/d/1U4DOZv5V9_7wJdqdicjp0tCmNIdRNJOc/view?usp=sharing) 
                         
## Notices
In [ShiftAdd](shiftadd/), we've gone beyond providing just the CUDA modules essential for the PyTorch operators that SW is utilizing. We've also extended our offerings to include a range of additional experimental CUDA packages. These involve enhancements such as incorporating position encodings into shift-wise operations and the addition of adaptive weights. Each package is accompanied by validation code for the CUDA operators, guaranteeing that their outputs are as anticipated. We're hopeful that this will be of assistance to you and would be grateful for your endorsements and references.

Moreover, the ERF component of SW is profoundly thought-provoking. Delving deeper into its analysis might yield a treasure trove of valuable information.
<p align="center">
<img src="erf\sw_00.png">
</p>

## Citation

If you use SW in your research, please consider the following BibTeX entry and giving us a star:
```BibTeX
@inproceedings{lidc2025sw,
  title={ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect},
  author={Dachong Li, Li Li, Zhuangzhuang Chen, Jianqiang Li},
  booktitle={CVPR},
  year={2025}
}
```
