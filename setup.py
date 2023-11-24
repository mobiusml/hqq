from setuptools import setup, find_packages

setup(
    name='hqq',
    version='1.0.0',    
    description='Half-Quadratic Quantization (HQQ)',
    url='https://github.com/mobiusml/hqq/tree/main/code',
    author='Dr. Hicham Badri',
    author_email='hicham@mobiuslabs.com',
    license='Apache 2',
    packages=['hqq', 'hqq/models', 'hqq/core'],
    install_requires=['numpy>=1.24.4','tqdm>=4.64.1', 'torch>=2.0.1', 'huggingface_hub', 'accelerate', 'timm', 'transformers[torch]'],
)
