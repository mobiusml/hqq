from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="hqq_aten",
    ext_modules=[
        CUDAExtension(
            "hqq_aten",
            [
                "hqq_aten_cuda.cpp",
                "hqq_aten_cuda_kernel.cu",
            ],
        )
    ],
    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
    cmdclass={"build_ext": BuildExtension},
)

# python3 setup_cuda.py install
