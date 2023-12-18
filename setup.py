from setuptools import setup, find_packages

setup(
    name='hqq',
    version='0.1.1',    
    description='Half-Quadratic Quantization (HQQ)',
    url='https://github.com/mobiusml/hqq/',
    author='Dr. Hicham Badri',
    author_email='hicham@mobiuslabs.com',
    license='Apache 2',
    packages=['hqq', 'hqq/core', 'hqq/engine', 'hqq/models', 'hqq/models/hf', 'hqq/models/timm', 'hqq/models/vllm'],
    install_requires=['numpy>=1.24.4','tqdm>=4.64.1', 'torch>=2.1.1', 'huggingface_hub', 'accelerate', 'timm', 'transformers>=4.36.1', 'termcolor'], #add vllm/langchain?
)
