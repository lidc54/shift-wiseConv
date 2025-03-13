# AddShift
References to [NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example.git) and [CudaDemo](https://github.com/YuxueYang1204/CudaDemo). Version v0 did not consider the organization of the data and prioritized verifying the correctness of the code.


## Environments
* NVIDIA Driver: 418.116.00
* CUDA: 11.0
* Python: 3
* PyTorch: 1.7.0+cu110
* GCC: 8.3.0


## Code structure
```shell
├── compare_module.py：分析多分支的函数在module里的表现
├── compare_mp_sp.py：分析多分支的函数在相同输入时的表现
├── compare_mp_linear.py：分析加权多分支的函数在相同输入时的表现，对比addshift_kernel_vmp
├── compare.py：分析**单分支的函数**的表现（与上一个实现对比）
├── cover：一重分支下的卷积输出利用情况（覆盖即利用，覆盖多次利用多次）注意在上一层还有多分支的分析
│   ├── hw14x14_R0.png
│   ├── hw14x14_R1.png
│   ├── hw28x28_R0.png
│   ├── hw28x28_R1.png
│   ├── hw56x56_R0.png
│   ├── hw56x56_R1.png
│   ├── hw7x7_R0.png
│   └── hw7x7_R1.png
├── cover.py：分析多分支时的卷积利用情况
├── is_jit_faster.py：分析**单分支的函数**能不能使用jit加速
├── LICENSE
├── ops
│   ├── __init__.py
│   ├── ops_py
│   │   ├── add_shift.py：加单分支/多分支函数加载为torch算子
│   │   └── __init__.py
│   └── src
│       ├── addshift_kernel_v0.cu：单分支函数
│       ├── addshift_ops_v0.cpp：单分支函数
│       ├── addshift_kernel_vmp.cu：多分支函数
│       ├── addshift_ops_vmp.cpp：多分支函数
│       ├── addshift_kernel_vmp_linear.cu：多分支函数, 多分支各有权重
│       ├── addshift_ops_vmp_linear.cpp：多分支函数, 多分支各有权重
│       ├── addshift_kernel_vmp_pos_em.cu：多分支函数, 多分支各有位置信息编码
│       └── addshift_ops_vmp_pos_em.cpp：多分支函数, 多分支各有位置信息编码
├── README.md
├── setup.py：单分支函数编译
├── SLaK_gr_sw_sf_se2.py：分析单分支的函数在module里的表现
├── SLaK_gr_sw_sf_se.py：分析单分支的函数在module里的表现
├── SLAK_se.py：分析单分支的函数在module里的表现，SLaK_gr_sw_sf_se2.py & SLaK_gr_sw_sf_se.py的总调用文件
└── test_ops.py：单分支函数耗时分析

```

**Setuptools**  
```shell
python setup.py install
```

### Run python
**Whether the test module runs smoothly or not**  
```shell
python test_ops.py
```

**To test the duration and correctness of the results of the module**  
```shell
python compare.py
```

**To test the duration and correctness of the results of the context modules**  
```shell
python SLAK_se.py
```

**To test whether torch.jit helps to accelerate**  
```shell
python is_jit_faster.py
```

## bug
- libc10.so: cannot open shared object file: No such file or directory
    - libc10.so是基于pytorch生成的，因此需要先导入torch包，然后再导入依赖于torch的包