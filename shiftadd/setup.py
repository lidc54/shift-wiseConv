from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="addshift",
    packages=find_packages(),
    version='0.1.0',
    # include_dirs=["include"],
    ext_modules=[
        # CUDAExtension(
        #     "add2",
        #     ["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
        # ),
        CUDAExtension(
            "addshift",
            ["./ops/src/addshift_ops_v0.cpp", "./ops/src/addshift_kernel_v0.cu"],
        ),
        CUDAExtension(
            "addshiftmp",
            ["./ops/src/addshift_ops_vmp.cpp", "./ops/src/addshift_kernel_vmp.cu"],
        ),
        CUDAExtension(
            "addshiftmp_linear",
            ["./ops/src/addshift_ops_vmp_linear.cpp", "./ops/src/addshift_kernel_vmp_linear.cu"],
        ),
        CUDAExtension(
            "addshiftmp_em",
            ["./ops/src/addshift_ops_vmp_pos_em.cpp", "./ops/src/addshift_kernel_vmp_pos_em.cu"],
        ),
        CUDAExtension(
            "addshiftmp_blur",
            ["./ops/src/addshift_ops_vmp_blur.cpp", "./ops/src/addshift_kernel_vmp_blur.cu"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
