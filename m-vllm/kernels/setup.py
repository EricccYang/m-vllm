from setuptools import setup, Extension
import pybind11
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension(
        "m_vllm_csrc",          # Python 模块名
        [
            "extension.cpp",     # C++ 源文件
            "attension.cu",      # CUDA 源文件
        ],
        extra_compile_args={
            "cxx": ["-std=c++17"],  # C++17 标准
            "nvcc": ["-std=c++17"], # CUDA 编译选项
        },
    ),
]


setup(
    name="m_vllm_csrc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.6",
)