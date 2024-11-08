# [shift-wise conv](https://github.com/lidc54/shift-wiseConv)
Our research finds that $3 \times 3$ convolutions can replace larger ones in CNNs, enhancing performance and echoing VGG's results. It also introduces novel parameter settings that have not been previously explored.


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


## based on repository [Sparse Large Kernel Network - SW](https://github.com/VITA-Group/SW)

 **name**   | **resolution** | **acc@1**       | **model**                                                                                          
:----------:|:--------------:|:---------------:|:--------------------------------------------------------------------------------------------------:
 **SW-T** | 224x224        | 83.39(300epoch)  | []() 
                         

## cite
If you find this repository useful, please consider giving a star star and cite our paper.

