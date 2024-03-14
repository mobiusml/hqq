from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="hqq_aten",
    ext_modules=[cpp_extension.CppExtension("hqq_aten", ["hqq_aten.cpp"])],
    extra_compile_args=["-O3"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

# python3 setup.py install
